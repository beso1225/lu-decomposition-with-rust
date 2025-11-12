use crate::matrix::{Matrix, MatrixState};

pub fn solve(mat: &Matrix, p: &Matrix, b: &Matrix) -> Matrix {
    if mat.get_state() != MatrixState::Lu {
        panic!("Matrix must be LU decomposed to solve linear systems");
    }

    if b.n() != mat.n() || b.m() != 1 {
        panic!("Right-hand side vector length must match matrix size");
    }
    let mut x = vec![0.0; mat.n()];
    let mut y = vec![0.0; mat.n()];
    // Apply permutation to b
    for i in 0..mat.n() {
        y[i] = b.get(p.get(i, 0) as usize, 0);
    }
    // Forward substitution to solve Ly = Pb
    for i in 0..mat.n() {
        for j in 0..i {
            y[i] -= mat.get(i, j) * y[j];
        }
    }

    // Backward substitution to solve Ux = y
    for i in (0..mat.n()).rev() {
        x[i] = y[i];
        for j in (i + 1)..mat.n() {
            x[i] -= mat.get(i, j) * x[j];
        }
        x[i] /= mat.get(i, i);
    }

    let mut result = Matrix::new(mat.n(), 1, true);
    for i in 0..mat.n() {
        result.set(i, 0, x[i]);
    }
    result
}