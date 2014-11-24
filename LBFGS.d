module LBFGS;

import std.math : abs, isnan, sqrt, sgn;
import std.stdio;
import std.range;


/*
Simple Limited Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) implementation.
Main reference for implementation:
"Numerical Optimization (2 ed.)" - J. Nocedal & S. Wright, Springer, 2006
Mostly chapters 7, 6 and 3.

For line search details, see also:
Moré, Jorge J., and David J. Thuente. "Line search algorithms with guaranteed
sufficient decrease." ACM Transactions on Mathematical Software (TOMS) 20.3
(1994): 286-307.

Want a job in large-scale ML? Please contact Benoit Rostykus:
firstname.lastname@adroll.com
*/
class LBFGS {

    ulong n; // problem's dimensionality
    ulong m; // rank of the inverse Hessian approximation
    float stop_eps; // stoping criterion on objective-function difference
    ulong max_it; // maximum number of outer-loop (ie steps taken) iterations
    ulong max_calls; // maximum number of calls to the func/grad eval delegate
    bool verbose; // print on stdout or not

    float[][] s; // s[k] = x(k+1) - x(k) => delta of solution progress
    float[][] y; // y[k] = grad(k+1) - grad(k) => delta of gradient progress
    float[] P; // stores current descent direction
    float[] alpha; // scalars used in 2-loop recursion
    float[] rho; // scalars 1/dot(y[k], s[k])
    float gamma; // approx H_{k,0} by gamma_k * I
    bool end_progress; // internal flag to indicate the solver can't go further

    // delegate evaluating objective-function and its gradient at a
    // given point x
    void delegate (const float[] x,
                   ref float[] resGrad,
                   out float resFunc) funcAndGrad;

    this(ulong n_,
         void delegate(const float[] x,
                   ref float[] resGrad,
                   out float resFunc) funcAndGrad_,
         ulong m_ = 5,
         bool verbose_ = true,
         float stop_eps_ = 1e-2,
         ulong max_calls_ = 100,
         ulong max_it_ = 40
        )
    {
        this.n = n_;
        this.m = m_;
        this.funcAndGrad = funcAndGrad_;
        this.stop_eps = stop_eps_;
        this.max_calls = max_calls_;
        this.max_it = max_it_;
        this.verbose = verbose_;
        this.end_progress = false;

        // mem allocations
        this.s = new float[][](m, n);
        this.y = new float[][](m, n);
        this.P = new float[n];
        this.alpha = new float[m];
        this.rho = new float[m];
        this.gamma = 0;

        // initialization
        reset();
    }

    void solve(ref float[] solution, out float func_val, float[] precond = null)
    {
        float[] X = new float[n];
        X[] = solution;
        float[] G = new float[n];
        G[] = 0;

        auto indC = cycle(iota(0, m));
        auto indCRev = cycle(retro(iota(0, m)));

        funcAndGrad(X, G, func_val);

        gamma = 1;
        float last_val = func_val;

        if(verbose)
        {
            writeln("[ It]\t[Calls]\tAlpha\tFunc val\tDiff last call");
            writeln("------------------------------------------------------");
            writefln("[  0]\t[  1]\t-\t%.4e", func_val);
        }
        ulong ind = 0;
        ulong calls = 1;
        while(ind < max_it && calls < max_calls)
        {
            /////// 1) Compute direction of descent /////////

            // two-loop recursion
            // P will contain H_k * Grad_k after the computation
            // also work when ind < m (will just do useless 0 + and x)
            P[] = G;
            foreach(ulong j; indCRev[ind..ind+m])
            {
                alpha[j] = rho[j] * dotProd(s[j], P);
                for(ulong i; i < n; ++i)
                    P[i] -= alpha[j] * y[j][i];
            }

            // multiplication H_{k, 0} * P
            if(precond !is null)
            {
                for(int ii = 0; ii < P.length; ++ii)
                    P[ii] *= precond[ii];
            }
            else
                P[] *= gamma;
            // second part of two-loops recursion
            foreach(ulong j; indC[ind..ind+m])
            {
                float beta = rho[j] * dotProd(y[j], P);
                for(ulong i; i < n; ++i)
                    P[i] += (alpha[j] - beta) * s[j][i];
            }

            /////// 2) Line search /////////////
            for(int i = 0; i < n; ++i)
                y[indC[ind]][i] -= G[i];
            float mu = line_search(X, G, P, last_val, func_val, calls);

            if(end_progress)
            {
                if(verbose)
                    writefln(
                        "[%3d]\t[%3d]\t%.3f\tCan't make any more progress.",
                        ind + 1, calls, mu);
                // revert last step
                for(int i = 0; i < n; ++i)
                    X[i] += mu * P[i];
                break;
            }
            ////// 3) Update vectors ///////////
            float delta = abs(last_val - func_val);
            last_val = func_val;
            if(verbose)
                writefln("[%3d]\t[%3d]\t%.3f\t%.4e \t%.3e",
                         ind + 1, calls, mu, func_val, delta);
            if(delta / abs(last_val) < stop_eps)
                break;
            for(int i = 0; i < n; ++i)
            {
                s[indC[ind]][i] = - mu * P[i];
                y[indC[ind]][i] += G[i];
            }
            float tmp = dotProd(y[indC[ind]], s[indC[ind]]);
            if(precond is null)
                gamma = tmp / dotProd(y[indC[ind]], y[indC[ind]]);
            rho[indC[ind]] = 1.0 / tmp;
            ind++;
        }
        solution = X;
    }

    float dummy_line_search(ref float[] X, ref float[] G,
                      ref float[] P,
                      ref float last_val,
                      ref float func_val,
                      ref ulong calls)
    {
        /* This dummy line search is for debugging purpose */
        float mu = 1;
        float last_mu = 0;
        while(calls < max_calls)
        {
            for(int i = 0; i < n; ++i)
                X[i] = X[i] - (mu - last_mu) * P[i];
            funcAndGrad(X, G, func_val);
            ++calls;
            if(func_val <= last_val)
                break;
            else
            {
                last_mu = mu;
                mu /= 2;
            }
        }
        return mu;
    }


    /* Try to find a point close to the unidimensional minimizer alpha_star of
       phi(x - alpha * p), which also satisfies the strong Wolfe conditions.
     */
    float line_search(ref float[] X, ref float[] G,
                      const float[] P,
                      ref float last_val,
                      ref float func_val,
                      ref ulong calls)
    {
        float mu = 1;

        float c1 = 1e-4;
        float c2 = 0.9;
        float max_mu = 1e3;
        float min_mu = 1e-10;

        float last_mu = 0;
        float last_phi_mu = last_val;
        ulong ix = 0;
        float phi_0 = last_val;
        float phi_prime_0 = -dotProd(P, G); // minus comes from the fact that P
                                            // is opposite sign of paper's one
        float last_phi_prime = phi_prime_0;
        float phi_prime = phi_prime_0;

        bool check_wolfe(float x, float f, float fp) nothrow @safe
        {
            if((f <= phi_0 + c1 * x * phi_prime_0) &&
               (f < last_val) &&
               (abs(fp) <= -c2 * phi_prime_0))
                return true;
            return false;
        }

        /* Returns the minimizer of a the cubic interpolation of phi fited
          on the func and derivative vals at low and high */
        float interpolate(float low, float high,
                          float phi_low, float phi_high,
                          float phi_prime_low, float phi_prime_high) pure
        {
            float d1 = (phi_prime_low + phi_prime_high
                        - 3 * (phi_low - phi_high) / (low - high));
            float d2 = (sgn(high - low)
                        * sqrt(d1 * d1 - phi_prime_low * phi_prime_high));
            float alpha_star = high - (high - low) * (
                (phi_prime_high + d2 - d1)/
                (phi_prime_high - phi_prime_low + 2*d2));
            return alpha_star;
        }


        float zoom(float low, float high,
                   float phi_low, float phi_high,
                   float phi_prime_low, float phi_prime_high)
        {
            if(abs(phi_high - phi_low)/abs(phi_low) < stop_eps)
                end_progress = true;
            if(calls == max_calls || end_progress ||
               mu < min_mu || mu > max_mu)
                return mu;
            last_mu = mu;
            mu = interpolate(low, high,
                             phi_low, phi_high,
                             phi_prime_low, phi_prime_high);
            if(isnan(mu))
            {
                // it means the cubic approximation doesn't have a minimizer
                // in the interval
                end_progress = true;
                return last_mu;
            }
            for(int i = 0; i < n; ++i)
                X[i] = X[i] - (mu - last_mu) * P[i];
            funcAndGrad(X, G, func_val);
            ++calls;
            last_phi_prime = phi_prime;
            phi_prime = -dotProd(P, G);
            // first Wolfe condition check
            if((func_val > phi_0 + c1 * mu * phi_prime_0) ||
               func_val >= phi_low)
                return zoom(low, mu,
                            phi_low, func_val,
                            phi_prime_low, phi_prime);
            else
            {
                // win! strong Wolfe conditions satisfied
                if(abs(phi_prime) <= -c2 * phi_prime_0)
                    return mu;
                else if(phi_prime * (high - low) >= 0)
                    return zoom(mu, low,
                                func_val, phi_low,
                                phi_prime, phi_prime_low);
                return zoom(mu, high,
                            func_val, phi_high,
                            phi_prime, phi_prime_high);
            }
        }

        while(calls < max_calls && mu > min_mu && mu < max_mu)
        {
            last_phi_mu = func_val;
            for(int i = 0; i < n; ++i)
                X[i] = X[i] - (mu - last_mu) * P[i];
            funcAndGrad(X, G, func_val);
            ++calls;
            last_phi_prime = phi_prime;
            phi_prime = -dotProd(P, G);
            // first Wolfe condition check
            if((func_val > phi_0 + c1 * mu * phi_prime_0) ||
                (ix > 0 && func_val >= last_val)) // fail! it's not a descent
                return zoom(last_mu, mu,
                            last_phi_mu, func_val,
                            last_phi_prime, phi_prime);
            // win! strong Wolfe conditions satisfied
            if(abs(phi_prime) <= -c2 * phi_prime_0)
                return mu;
            else if(phi_prime >= 0)
                return zoom(mu, last_mu,
                            func_val, last_phi_mu,
                            phi_prime, last_phi_prime);
            last_mu = mu;
            mu *= 2; // we could use an extrapolation of cubic approx instead
            ++ix;
        }
        end_progress = true;
        return mu;
    }

    float dotProd(const float[] z1, ref float[] z2) pure nothrow @safe
    {
        double res = 0;
        for(int i = 0; i < z1.length; ++i)
            res += z1[i] * z2[i];
        return res;
    }

    void reset()
    {
        for(int k = 0; k < m; ++k)
        {
            s[k][] = 0;
            y[k][] = 0;
        }
        P[] = 0;
        alpha[] = 0;
        rho[] = 1;
    }
}

unittest {
    bool verbose = false;
    writeln("[TEST] Testing against quadratic function.");
    // x -> diag(1,1,1,1,...,7,7,7,7)||x - (3,3,3,...,2,2,2)||^2 + 4
    void test(const float[] x, ref float[] res_grad, out float res_func) pure
    {
        ulong d = x.length;
        res_func = 4;
        for(int i = 0; i < d; ++i)
        {
            if(i < d / 2)
            {
                res_grad[i] = 2 * (x[i] - 3);
                res_func += (x[i] - 3) * (x[i] - 3);
            }
            else
            {
                res_grad[i] = 2 * 7 * (x[i] - 2);
                res_func += 7 * (x[i] - 2) * (x[i] - 2);
            }
        }
    }

    write("[*] Basic initial conditions... ");
    auto d = 10;
    auto m = 4;
    auto solver = new LBFGS(d, &test, m, verbose, 1e-5);
    float[] solution;
    solution.length = d;
    solution[] = 1;
    float val;
    solver.solve(solution, val);

    assert(abs(val - 4) < 1e-2);
    int i = 0;
    while(i < d)
    {
        if(i < d/2)
            assert(abs(solution[i] - 3) < 1e-3);
        else
            assert(abs(solution[i] - 2) < 1e-3);
        ++i;
    }
    writeln("Ok.");

    write("[*] Harder initial conditions and lower hessian rank... ");
    d = 100;
    m = 3;
    solver = new LBFGS(d, &test, m, verbose, 1e-1);
    solution.length = d;
    i = 0;
    while(i < d)
    {
        if(i < d/4)
            solution[i] = -20;
        else if(i >= d/4 && i < d/2)
            solution[i] = 57;
        else if(i >= d/2 && i < 3*d/4)
            solution[i] = -6;
        else
            solution[i] = 13;
        ++i;
    }
    solver.solve(solution, val);

    assert(abs(val - 4) < 1e-2);
    i = 0;
    while(i < d)
    {
        if(i < d/2)
            assert(abs(solution[i] - 3) < 1e-1);
        else
            assert(abs(solution[i] - 2) < 1e-1);
        ++i;
    }
    writeln("Ok.");
}

unittest {
    import std.math : pow;
    bool verbose = false;
    writeln("[TEST] Testing against Rosenbrock function.");
    // x -> sum_{i=1}^{d/2}100(x_{2i}-x_{2i-1}^2)^2 + (1-x_{2i-1})^2
    // solution is at (1,1,1,...,1)
    void rosenbrock(const float[] x,
                    ref float[] res_grad, out float res_func) pure
    {
        ulong d = x.length;
        res_func = 0;
        for(int i = 0; i < d / 2; ++i)
        {
            res_func += 100 * pow(x[2*i + 1] - x[2*i] * x[2*i], 2);
            res_func += pow(1 - x[2*i], 2);

            res_grad[2 * i] = (
                400 * pow(x[2*i], 3)
                - 400 * x[2*i] * x[2*i + 1]
                -2 * (1 - x[2*i]));
            res_grad[2*i + 1] = 200 * x[2*i + 1] - 200 * pow(x[2*i], 2);
        }
    }

    write("[*] Easy starting point... ");
    auto d = 30;
    auto m = 10;
    auto solver = new LBFGS(d, &rosenbrock, m, verbose, 1e-3);
    float[] solution;
    solution.length = d;
    solution[] = 0.9;
    float val;
    solver.solve(solution, val);

    assert(abs(val - 0) < 1e-2);
    foreach(w; solution)
        assert(abs(w - 1) < 5e-2);
    writeln("Ok.");
}
