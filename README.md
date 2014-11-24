# LBFGS-D

**LBFGS-D** provides a native implementation of the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno algorithm
(L-BFGS) in the **D** programming language.

The implementation is meant to be as simple and readable as possible, as a single self-contained file module.

It doesn't rely on any third-party dependency and follows closely Nocedal & Wright's algorithm description provided
in [Numerical Optimization (2 ed. Springer, 2006)](http://www.springer.com/mathematics/book/978-0-387-30303-1) book.
It is particularly well suited for cases where the function & gradient evaluations are relatively expensive compared
to the actual quasi-Newton variables updates.

The API requires you to specify a:
```D
void delegate(const float[] x, ref float[] resGrad, out float resFunc)
```
which computes at point `x` the value `resFunc` of the function to minimize,
as well as its gradient `resGrad` at point `x`.

```D
import LBFGS : LBFGS;
auto d = 30; // 30 dimensions
auto m = 5; // rank of inv-Hessian approximation
void my_function(const float[] x, ref float[] res_grad, out float res_func)
{
  ...
}
auto solution = new float[d];
solution[] = 0; // specify your starting point here
float min_val;
auto lbfgs = new LBFGS(d, &my_function, m); // will allocate memory
lbfgs.solve(solution, min_val); // run optimizer
```
Additionaly, one can specify a Jacobi preconditioner to be used.

More examples of usage can be found in the module unit-tests:
```D
rdmd -main -unittest LBFGS.d
```
