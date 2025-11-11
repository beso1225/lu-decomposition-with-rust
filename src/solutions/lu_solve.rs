use crate::matrix::{DenseMatrix, MatrixState};

pub fn solve(mat: &DenseMatrix, p: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    if mat.state != MatrixState::Lu {
        panic!("Matrix must be LU decomposed to solve linear systems");
    }

    if b.n != mat.n {
        panic!("Right-hand side vector length must match matrix size");
    }
    let mut x = vec![0.0; mat.n];
    let mut y = vec![0.0; mat.n];

    // Apply permutation to b
    for i in 0..mat.n {
        y[i] = b.data[p.data[i][0] as usize][0];
    }
    // Forward substitution to solve Ly = Pb
    for i in 0..mat.n {
        for j in 0..i {
            y[i] -= mat.data[i][j] * y[j];
        }
    }

    // Backward substitution to solve Ux = y
    for i in (0..mat.n).rev() {
        x[i] = y[i];
        for j in (i + 1)..mat.n {
            x[i] -= mat.data[i][j] * x[j];
        }
        x[i] /= mat.data[i][i];
    }

    let mut result = DenseMatrix::new(mat.n, 1);
    for i in 0..mat.n {
        result.data[i][0] = x[i];
    }
    result
}