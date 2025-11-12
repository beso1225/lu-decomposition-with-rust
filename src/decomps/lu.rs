use crate::matrix::{Matrix, MatrixState};

impl Matrix {
    pub fn lu_decomposition(&mut self) -> Matrix { // Changes original matrix to L and U matrix and returns permutation vector
        if self.n() != self.m() {
            panic!("Matrix must be square for LU decomposition");
        }
        let mut p = Matrix::new(self.n(), 1, true);
        for i in 0..self.n() {
            p.set(i, 0, i as f64);
        }
        for k in 0..(self.n() - 1) {
            let max_row = self.max_abs_in_column(k, k);
            if max_row != k {
                self.swap_rows(k, max_row);
                p.swap_rows(k, max_row);
            }
            let pivot_value_inverse = 1.0 / self.get(k, k);
            for i in (k + 1)..self.n() {
                let factor = self.get(i, k) * pivot_value_inverse;
                self.set(i, k, factor);
                for j in (k + 1)..self.m() {
                    let value = self.get(i, j) - self.get(i, k) * self.get(k, j);
                    self.set(i, j, value);
                }
            }
            // println!("After processing column {}:", k);
            // mat.show();
        }
        self.set_state(MatrixState::Lu);
        p
    }
}
