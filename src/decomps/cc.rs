use std::collections::HashSet;

use crate::matrix::{Matrix, MatrixState};

impl Matrix {
    pub fn cc_decomposition(&self) -> Matrix {
        if self.get_state() == MatrixState::Cc {
            println!("Matrix is already CC decomposed");
            return self.copy()
        }
        if self.is_symmetric_positive_definite() == false {
            panic!("Matrix must be symmetric positive definite for Cholesky-Crout decomposition");
        }
        let n = self.n();
        let mut l = Matrix::new(n, n, true);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    let diag_value = self.get(i, i) - sum;
                    if diag_value <= 0.0 {
                        panic!("Matrix is not positive definite");
                    }
                    l.set(i, j, diag_value.sqrt());
                } else {
                    let value = (self.get(i, j) - sum) / l.get(j, j);
                    l.set(i, j, value);
                }
            }
        }
        l.set_state(MatrixState::Cc);
        l
    }

    pub fn incomplete_cc_decomposition(&mut self, k: usize) -> Matrix {
        match self {
            Matrix::Dense(_) => self.change(),
            Matrix::Sparse(_) => {}
        }
        // Build pattern (symmetric) of allowed nonzeros for L
        let n = self.n();
        let mut l = Matrix::new(n, n, false); // build L as sparse
        let mut pattern: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        // initialize pattern from A (ensure symmetry)
        for i in 0..n {
            for j in 0..n {
                if self.get(i, j) != 0.0 || i == j {
                    pattern[i].insert(j);
                    pattern[j].insert(i);
                }
            }
        }

        // expand pattern k times (propagate fill-in) keeping symmetry
        for _iter in 0..k {
            let mut new_pattern = pattern.clone();
            for i in 0..n {
                let row_snapshot: Vec<usize> = pattern[i].iter().cloned().collect();
                for &j in &row_snapshot {
                    if i == j { continue; }
                    let col_snapshot: Vec<usize> = pattern[j].iter().cloned().collect();
                    for &s in &col_snapshot {
                        if s == i { continue; }
                        new_pattern[i].insert(s);
                        new_pattern[s].insert(i);
                    }
                }
            }
            pattern = new_pattern;
        }

        // Compute incomplete Cholesky-Crout using the pattern (only entries in pattern are computed)
        for i in 0..n {
            // sort columns for deterministic order
            let mut cols: Vec<usize> = pattern[i].iter().cloned().collect();
            cols.sort_unstable();
            for &j in &cols {
                if j > i { continue; }

                // sum over common s in pattern[i] and pattern[j] with s < j
                let common_s: Vec<usize> = pattern[i]
                    .iter()
                    .cloned()
                    .filter(|&s| s < j && pattern[j].contains(&s))
                    .collect();
                let mut sum = 0.0;
                for s in common_s {
                    sum += l.get(i, s) * l.get(j, s);
                }

                if i == j {
                    let diag_value = self.get(i, i) - sum;
                    if diag_value <= 0.0 {
                        panic!("Matrix is not positive definite (or pattern too restrictive)");
                    }
                    l.set(i, j, diag_value.sqrt());
                } else {
                    let denom = l.get(j, j);
                    if denom == 0.0 {
                        // can't divide; leave as zero (pattern may be too restrictive) or panic
                        panic!("Zero diagonal encountered during IC factorization");
                    }
                    let value = (self.get(i, j) - sum) / denom;
                    l.set(i, j, value);
                }
            }
        }

        l.set_state(MatrixState::ICc);
        l
    }
}

mod test{
    #[test]
    fn test_cc_decomposition() {
        use crate::matrix::Matrix;
        let mut mat = Matrix::new(3, 3, true);
        let mat_original_data = vec![
            vec![4.0, 2.0, 2.0],
            vec![2.0, 3.0, 1.0],
            vec![2.0, 1.0, 3.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                mat.set(i, j, mat_original_data[i][j]);
            }
        }
        println!("Original Matrix:");
        mat.show();
        let cc_mat = mat.cc_decomposition();
        println!("CC Decomposed Matrix:");
        cc_mat.show();
    }

    #[test]
    fn test_incomplete_cc_decomposition() {
        use crate::matrix::Matrix;

        // Build a larger sparse SPD matrix that will induce fill-in under elimination.
        // We'll construct a graph with a cycle (non-chordal) so incomplete Cholesky with
        // limited pattern will gain fill-in as k increases.
        let n = 12usize;
        let mut mat = Matrix::new(n, n, true);

        // edges: an 8-cycle among nodes 0..7
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..8 {
            edges.push((i, (i + 1) % 8));
        }
        // some extra links to connect remaining nodes and create elimination interactions
        edges.push((2, 9));
        edges.push((5, 10));
        edges.push((9, 11));
        edges.push((10, 11));

        // set off-diagonals to -1.0 for each undirected edge
        for &(i, j) in &edges {
            mat.set(i, j, -1.0);
            mat.set(j, i, -1.0);
        }

        // set diagonal to be strictly diagonally dominant (sum abs off-diagonals + 1)
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += mat.get(i, j).abs();
            }
            mat.set(i, i, row_sum + 1.0);
        }

        println!("Original sparse SPD Matrix (n={}) :", n);
        mat.show();

        let mut nnz_counts: Vec<usize> = Vec::new();
        for k in 0..4 {
            let l = mat.incomplete_cc_decomposition(k);
            // count nonzeros in L
            let mut nnz = 0usize;
            for i in 0..n {
                for j in 0..n {
                    if l.get(i, j) != 0.0 {
                        nnz += 1;
                    }
                }
            }
            println!("k = {} -> ICc L nonzeros = {}", k, nnz);
            l.show();
            nnz_counts.push(nnz);
        }

        // Expect that allowing more fill-in increases or at least does not decrease nnz.
        // Specifically, for this non-chordal graph we expect nnz_counts[1] > nnz_counts[0].
        assert!(nnz_counts[1] > nnz_counts[0], "Expected more nonzeros for k=1 than k=0");
        assert!(nnz_counts[2] >= nnz_counts[1], "Expected non-decreasing nnz with k");
    }

    #[test]
    fn test_cc_on_sparse_spd_matrix() {
        use crate::matrix::Matrix;

        // Construct the same sparse SPD matrix used in test_incomplete_cc_decomposition
        let n = 12usize;
        let mut mat = Matrix::new(n, n, true);
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..8 {
            edges.push((i, (i + 1) % 8));
        }
        edges.push((2, 9));
        edges.push((5, 10));
        edges.push((9, 11));
        edges.push((10, 11));
        for &(i, j) in &edges {
            mat.set(i, j, -1.0);
            mat.set(j, i, -1.0);
        }
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += mat.get(i, j).abs();
            }
            mat.set(i, i, row_sum + 1.0);
        }

        // compute full Cholesky (CC)
        let l = mat.cc_decomposition();
        l.show();

        // build L^T by transposing
        let mut lt = Matrix::new(n, n, true);
        for i in 0..n {
            for j in 0..n {
                lt.set(j, i, l.get(i, j));
            }
        }

        // compute L * L^T
        let prod = Matrix::product(&l, &lt);

        // difference norm
        let diff = Matrix::minus(&prod, &mat);
        let err = diff.norm_inf();
        println!("CC reconstruction max row-sum error: {}", err);
        assert!(err < 1e-8, "Cholesky reconstruction error too large: {}", err);
    }
}