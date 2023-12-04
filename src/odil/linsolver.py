import numpy as np


def solve(matr, rhs, args, status=None, linsolver="direct"):
    import scipy.sparse
    import scipy.sparse.linalg

    if args.linsolver_maxiter is None:
        if args.linsolver == 'lsqr':
            args.linsolver_maxiter = 1000
        else:
            args.linsolver_maxiter = 50

    def get_sparse_eye(size):
        return scipy.sparse.diags(np.ones(size), format='csr')

    eye = get_sparse_eye(matr.shape[1])
    matr_reg = matr.T.dot(matr).tocsr()
    if args.linsolver_damp:
        matr_reg += args.linsolver_damp**2 * eye
    if args.linsolver_dampdiag:
        matr_reg += args.linsolver_dampdiag**2 * scipy.sparse.diags(
            matr_reg.diagonal())
    rhs_reg = matr.T.dot(rhs)
    if linsolver == "direct":
        # Normal equations.
        sol = scipy.sparse.linalg.spsolve(matr_reg,
                                          rhs_reg,
                                          permc_spec='MMD_ATA')
    elif linsolver == "directsq":
        # Original system, assuming a square matrix.
        sol = scipy.sparse.linalg.spsolve(matr, rhs, permc_spec='MMD_ATA')
    elif linsolver == "direct_cu":
        import cupy
        import cupyx.scipy.sparse
        import cupyx.scipy.sparse.linalg
        matr_reg = cupyx.scipy.sparse.csr_matrix(matr_reg)
        rhs_reg = cupy.array(rhs_reg)
        sol = cupyx.scipy.sparse.linalg.spsolve(matr_reg, rhs_reg)
        sol = sol.get()
    elif linsolver == "sparseqr":
        import sparseqr
        sol = sparseqr.solve(matr, rhs, tolerance=args.linsolver_tol)
    elif linsolver == "lsqr":
        sol, _, itn, _, _, anorm, acond, arnorm = \
                scipy.sparse.linalg.lsqr(
            matr,
            rhs,
            damp=args.linsolver_damp,
            atol=args.linsolver_tol,
            btol=args.linsolver_tol,
            iter_lim=args.linsolver_maxiter)[:8]
        status['residual'] = arnorm
        status['anorm'] = anorm
        status['acond'] = acond
        status['niter'] = itn
    elif linsolver == "lsqr_cu":
        if cupy is None:
            raise ModuleNotFoundError(
                "Module CuPy not found. Install with 'pip install cupy-cuda110'"
            )
        # XXX cupy does not support non-square matrices
        sol = cupyx.scipy.sparse.linalg.lsqr(matr, rhs)[0]
    elif linsolver == "multigrid":
        import pyamg
        if pyamg is None:
            raise ModuleNotFoundError(
                "Module PyAMG not found. Install with 'pip install pyamg'")
        ml = pyamg.smoothed_aggregation_solver(matr_reg)
        residuals = []
        sol = ml.solve(b=rhs_reg,
                       tol=args.linsolver_tol,
                       residuals=residuals,
                       accel='cg',
                       maxiter=args.linsolver_maxiter)
        status['residual'] = residuals[-1]
        status['niter'] = len(residuals)
    elif linsolver == "bicgstab":
        residuals = []

        def callback(x):
            residuals.append(np.mean((matr_reg.dot(x) - rhs_reg)**2)**0.5)

        sol, _ = scipy.sparse.linalg.bicgstab(matr_reg,
                                              rhs_reg,
                                              tol=0,
                                              atol=args.linsolver_tol,
                                              callback=callback,
                                              maxiter=args.linsolver_maxiter)
        status['residual'] = residuals[-1]
        status['niter'] = len(residuals)
    else:
        raise ValueError("Unknown linsolver=" + linsolver)

    return sol


def add_arguments(parser):
    parser.add_argument('--linsolver',
                        type=str,
                        choices=[
                            "multigrid",
                            "direct",
                            "directsq",
                            "direct_cu",
                            "sparseqr",
                            "lsqr",
                            "lsqr_cu",
                            "bicgstab",
                        ],
                        default="direct",
                        help="Linear solver to use")
    parser.add_argument('--linsolver_maxiter',
                        type=int,
                        default=None,
                        help="Maximum number of iterations of linear solver")
    parser.add_argument('--linsolver_tol',
                        type=float,
                        default=1e-6,
                        help="Tolerance for linear solver")
    parser.add_argument('--linsolver_damp',
                        type=float,
                        default=0,
                        help="Relaxation factor (0: no relaxation)")
    parser.add_argument('--linsolver_dampdiag',
                        type=float,
                        default=0,
                        help="Multiplier for diagonal (0: no relaxation)")
    parser.add_argument('--linsolver_verbose',
                        type=int,
                        default=0,
                        help="Verbosity level for linsolver messages")
    parser.add_argument('--linsolver_history',
                        type=int,
                        default=0,
                        help="Dump history from linsolver status")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--nlvl',
                        type=int,
                        default=100,
                        help="Multigrid levels")
    parser.add_argument('--smooth_pre',
                        type=int,
                        default=2,
                        help="Pre-smoothing steps")
    parser.add_argument('--smooth_post',
                        type=int,
                        default=2,
                        help="Post-smoothing steps")
    parser.add_argument('--omega',
                        type=float,
                        default=0.6,
                        help="Jacobi smoother relaxation factor")
    parser.add_argument('--ndirect',
                        type=int,
                        default=3,
                        help="Systems on smaller grids "
                        "are solved with direct solver")
    parser.add_argument('--restriction',
                        type=str,
                        choices=('full', 'half', 'injection'),
                        default='full',
                        help="Multigrid restriction type")
