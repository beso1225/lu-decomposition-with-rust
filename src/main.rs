mod solutions;
use solutions::{lu_solve, cg};
mod decomps;
use decomps::lu;
mod matrix;
use matrix::Matrix;

fn main() {
    // let mut mat = Matrix::new(3, 3);
    // mat.data = vec![
    //     vec![2.0, 1.0, 1.0],
    //     vec![4.0, -6.0, 0.0],
    //     vec![-2.0, 7.0, 2.0],
    // ];
    let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input2.csv");
    println!("Original Matrix:");
    mat.show();
    println!("Right-hand side Vector b: {:?}", b);
    let p = lu::lu_decomposition(&mut mat);
    println!("LU Decomposed Matrix:");
    mat.show();
    println!("Permutation Vector:");
    p.show();
    // let b = vec![5.0, -2.0, 9.0];
    if b.is_empty() {
        println!("No right-hand side vector provided.");
        return;
    }
    let x = lu_solve::lu_solve(&mat, &p, &b);
    println!("Solution Vector x:");
    x.show();

    // Example usage of Conjugate Gradient solver
    let (mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input2.csv");
    let initial_guess = None; // or Some(Matrix::new(mat.n, 1)) for a zero initial guess
    let max_iter = 1000;
    let x_cg = cg::cg_solve(&mat, &b, initial_guess, max_iter);
    println!("CG Solution Vector x:");
    x_cg.show();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output_vector(filename: &str) -> Matrix {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(filename)
            .expect("Cannot open output CSV file");

        let mut vals: Vec<f64> = Vec::new();
        for result in reader.records() {
            let record = result.expect("Error reading csv record");
            for field in record.iter() {
                let s = field.trim();
                if s.is_empty() { continue; }
                if let Ok(num) = s.parse::<f64>() {
                    vals.push(num);
                } else {
                    panic!("Failed to parse '{}' as f64 in {}", s, filename);
                }
            }
        }
        let n = vals.len();
        let mut matrix = Matrix::new(n, 1);
        for i in 0..n {
            matrix.data[i][0] = vals[i];
        }
        matrix
    }

    #[test]
    fn test_solve_10x10() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input2.csv");
        let p = lu::lu_decomposition(&mut mat);
        let x = lu_solve::lu_solve(&mat, &p, &b);
        let expected_x = output_vector("src/test/output/output2.csv");
        for i in 0..x.n {
            assert!((x.data[i][0] - expected_x.data[i][0]).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.data[i][0], x.data[i][0]);
        }
    }

    #[test]
    fn test_solve_50x50() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input3.csv");
        let p = lu::lu_decomposition(&mut mat);
        let x = lu_solve::lu_solve(&mat, &p, &b);
        let expected_x = output_vector("src/test/output/output3.csv");
        for i in 0..x.n {
            assert!((x.data[i][0] - expected_x.data[i][0]).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.data[i][0], x.data[i][0]);
        }
    }

    #[test]
    fn test_solve_50x50_with_cg() {
        let (mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input3.csv");
        let initial_guess = None;
        let max_iter = 1000;
        let x = cg::cg_solve(&mat, &b, initial_guess, max_iter);
        let expected_x = output_vector("src/test/output/output3.csv");
        for i in 0..x.n {
            assert!((x.data[i][0] - expected_x.data[i][0]).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.data[i][0], x.data[i][0]);
        }
    }
}
