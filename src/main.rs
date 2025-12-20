mod solutions;
use solutions::{lu_solve, cg};
mod decomps;
mod matrix;
use matrix::{Matrix};

fn main() {
    // let mut mat = Matrix::new(3, 3);
    // mat.data = vec![
    //     vec![2.0, 1.0, 1.0],
    //     vec![4.0, -6.0, 0.0],
    //     vec![-2.0, 7.0, 2.0],
    // ];
    let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("tests/input/input2.csv", true);
    println!("Original Matrix:");
    mat.show();
    println!("Right-hand side Vector b: {:?}", b);
    let p = mat.lu_decomposition();
    println!("LU Decomposed Matrix:");
    mat.show();
    println!("Permutation Vector:");
    p.show();
    // let b = vec![5.0, -2.0, 9.0];
    if b.is_empty() {
        println!("No right-hand side vector provided.");
        return;
    }
    let x = lu_solve::solve(&mat, &p, &b);
    println!("Solution Vector x:");
    x.show();

    // Example usage of Conjugate Gradient solver
    let (mat, b) = Matrix::read_from_csv_with_right_hand_side("tests/input/input2.csv", true);
    let initial_guess = None; // or Some(Matrix::new(mat.n(), 1, true)) for a zero initial guess
    let max_iter = 1000;
    let mut x_cg = cg::solve(&mat, &b, initial_guess, max_iter);
    println!("CG Solution Vector x:");
    x_cg.show();
    x_cg.change();
    println!("CG Solution Vector as Sparse Matrix:");
    x_cg.show();
}
