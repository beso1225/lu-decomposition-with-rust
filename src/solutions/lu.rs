use crate::matrix::{Matrix, MatrixState};

pub fn lu_decomposition(mat: &mut Matrix) -> Matrix { // Changes original matrix to L and U matrix and returns permutation vector
    if mat.n != mat.m {
        panic!("Matrix must be square for LU decomposition");
    }
    let mut p = Matrix::new(mat.n, 1);
    for i in 0..mat.n {
        p.data[i][0] = i as f64;
    }
    for k in 0..(mat.n - 1) {
        let max_row = mat.max_abs_in_column(k, k);
        if max_row != k {
            mat.data.swap(k, max_row);
            p.data.swap(k, max_row);
        }
        let pivot_value_inverse = 1.0 / mat.data[k][k];
        for i in (k + 1)..mat.n {
            mat.data[i][k] *= pivot_value_inverse;
            for j in (k + 1)..mat.m {
                mat.data[i][j] -= mat.data[i][k] * mat.data[k][j];
            }
        }
        // println!("After processing column {}:", k);
        // mat.show();
    }
    mat.state = MatrixState::Lu;
    p
}

pub fn solve_linear_system(mat: &Matrix, p: &Matrix, b: &Matrix) -> Matrix {
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

    let mut result = Matrix::new(mat.n, 1);
    for i in 0..mat.n {
        result.data[i][0] = x[i];
    }
    result
}