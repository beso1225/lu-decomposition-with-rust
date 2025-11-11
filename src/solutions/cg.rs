use std::collections::HashMap;

use crate::matrix::Matrix;
use crate::decomps::cc;

pub fn solve(mat: &Matrix, b: &Matrix, initial_guess: Option<Matrix>, max_iter: usize) -> Matrix {
    // Simple CG implementation, with no preconditioning
    if !mat.is_symmetric_positive_definite() {
        panic!("Matrix must be symmetric positive definite for Conjugate Gradient method");
    }
    if b.n != mat.n {
        panic!("Right-hand side vector length must match matrix size");
    }

    let mut x = match initial_guess {
        Some(guess) => {
            if guess.n != mat.n || guess.m != 1 {
                panic!("Initial guess vector has incorrect dimensions");
            }
            let mut map = HashMap::new();
            map.insert(0, guess.copy());
            map
        },
        None => {
            let mut map = HashMap::new();
            map.insert(0, Matrix::new(mat.n, 1));
            map
        }
    };

    // CG algorithm implementation goes here"
    println!("Starting Conjugate Gradient Solver...");
    let mut r = HashMap::new();
    let mut p = HashMap::new();
    r.insert(0, Matrix::minus(b, &Matrix::product(mat, x.get(&0).unwrap())));
    p.insert(0, r.get(&0).unwrap().copy());

    for k in 0..max_iter {
        let r_k = r.get(&k).unwrap().copy();
        let p_k = p.get(&k).unwrap().copy();

        let mat_p_k = Matrix::product(mat, &p_k);
        let p_k_inner = Matrix::inner_product(&p_k, &mat_p_k);
        if p_k_inner.abs() < 1e-10 {
            break; // Avoid division by zero
        }
        let alpha_k = Matrix::inner_product(&r_k, &r_k) / p_k_inner;

        let x_next = Matrix::plus(&x.get(&k).unwrap(), &p_k.scalar_multiply(alpha_k));
        x.insert(k + 1, x_next.copy());
        let r_next = Matrix::minus(&r_k, &mat_p_k.scalar_multiply(alpha_k));
        r.insert(k + 1, r_next.copy());

        let beta_k = - (Matrix::inner_product(&r_next, &mat_p_k) / p_k_inner);

        let p_next = Matrix::plus(&r_next, &p_k.scalar_multiply(beta_k));
        p.insert(k + 1, p_next.copy());

        if r_next.norm_inf() < 1e-10 {
            break; // Converged
        }
    }
    println!("Conjugate Gradient Solver finished. Last iteration: {}", x.len() - 1);
    x.get(&(x.len() - 1)).unwrap().copy()
}

pub fn pcg_solve_cc(mat: &Matrix, p: &Matrix, b: &Matrix) -> Matrix {
    unimplemented!()
}