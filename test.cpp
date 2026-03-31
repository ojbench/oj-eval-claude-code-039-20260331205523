#include "CSRMatrix.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace sjtu;

void test_basic_construction() {
    cout << "Testing basic construction..." << endl;
    CSRMatrix<int> mat(3, 4);
    assert(mat.getRowSize() == 3);
    assert(mat.getColSize() == 4);
    assert(mat.getNonZeroCount() == 0);
    cout << "PASSED!" << endl;
}

void test_set_get() {
    cout << "Testing set and get..." << endl;
    CSRMatrix<int> mat(4, 5);

    mat.set(0, 1, 5);
    mat.set(0, 3, 7);
    mat.set(1, 2, 3);
    mat.set(2, 0, 9);
    mat.set(3, 4, 2);

    assert(mat.get(0, 1) == 5);
    assert(mat.get(0, 3) == 7);
    assert(mat.get(1, 2) == 3);
    assert(mat.get(2, 0) == 9);
    assert(mat.get(3, 4) == 2);
    assert(mat.get(0, 0) == 0);  // Should return default (0)
    assert(mat.getNonZeroCount() == 5);

    // Test updating existing element
    mat.set(0, 1, 10);
    assert(mat.get(0, 1) == 10);
    assert(mat.getNonZeroCount() == 5);  // Count should not change

    cout << "PASSED!" << endl;
}

void test_dense_matrix_conversion() {
    cout << "Testing dense matrix conversion..." << endl;

    vector<vector<int>> dense = {
        {0, 1, 0, 2},
        {3, 0, 0, 0},
        {0, 0, 4, 5}
    };

    CSRMatrix<int> mat(3, 4, dense);
    assert(mat.getNonZeroCount() == 5);
    assert(mat.get(0, 1) == 1);
    assert(mat.get(0, 3) == 2);
    assert(mat.get(1, 0) == 3);
    assert(mat.get(2, 2) == 4);
    assert(mat.get(2, 3) == 5);

    // Convert back to dense
    auto result = mat.getMatrix();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(result[i][j] == dense[i][j]);
        }
    }

    cout << "PASSED!" << endl;
}

void test_matrix_vector_multiplication() {
    cout << "Testing matrix-vector multiplication..." << endl;

    vector<vector<int>> dense = {
        {1, 0, 2},
        {0, 3, 0},
        {4, 0, 5}
    };

    CSRMatrix<int> mat(3, 3, dense);
    vector<int> vec = {1, 2, 3};

    auto result = mat * vec;
    assert(result.size() == 3);
    assert(result[0] == 1*1 + 0*2 + 2*3);  // 7
    assert(result[1] == 0*1 + 3*2 + 0*3);  // 6
    assert(result[2] == 4*1 + 0*2 + 5*3);  // 19

    cout << "PASSED!" << endl;
}

void test_row_slice() {
    cout << "Testing row slice..." << endl;

    vector<vector<int>> dense = {
        {1, 0, 2},
        {0, 3, 0},
        {4, 0, 5},
        {0, 6, 0}
    };

    CSRMatrix<int> mat(4, 3, dense);
    auto slice = mat.getRowSlice(1, 3);

    assert(slice.getRowSize() == 2);
    assert(slice.getColSize() == 3);
    assert(slice.getNonZeroCount() == 3);

    assert(slice.get(0, 1) == 3);  // Was row 1
    assert(slice.get(1, 0) == 4);  // Was row 2
    assert(slice.get(1, 2) == 5);  // Was row 2

    cout << "PASSED!" << endl;
}

void test_exception_handling() {
    cout << "Testing exception handling..." << endl;

    CSRMatrix<int> mat(3, 3);

    // Test invalid_index on get
    try {
        mat.get(5, 0);
        assert(false);  // Should not reach here
    } catch (const invalid_index&) {
        // Expected
    }

    // Test invalid_index on set
    try {
        mat.set(0, 5, 10);
        assert(false);  // Should not reach here
    } catch (const invalid_index&) {
        // Expected
    }

    // Test size_mismatch on multiplication
    vector<int> wrong_size_vec = {1, 2};
    try {
        mat * wrong_size_vec;
        assert(false);  // Should not reach here
    } catch (const size_mismatch&) {
        // Expected
    }

    // Test invalid_index on row slice
    try {
        mat.getRowSlice(2, 5);
        assert(false);  // Should not reach here
    } catch (const invalid_index&) {
        // Expected
    }

    cout << "PASSED!" << endl;
}

void test_csr_constructor() {
    cout << "Testing CSR format constructor..." << endl;

    // Create a simple CSR matrix manually
    vector<size_t> indptr = {0, 2, 3, 5};
    vector<size_t> indices = {0, 2, 1, 0, 2};
    vector<int> data = {1, 2, 3, 4, 5};

    CSRMatrix<int> mat(3, 3, 5, indptr, indices, data);

    assert(mat.getRowSize() == 3);
    assert(mat.getColSize() == 3);
    assert(mat.getNonZeroCount() == 5);

    assert(mat.get(0, 0) == 1);
    assert(mat.get(0, 2) == 2);
    assert(mat.get(1, 1) == 3);
    assert(mat.get(2, 0) == 4);
    assert(mat.get(2, 2) == 5);

    cout << "PASSED!" << endl;
}

int main() {
    try {
        test_basic_construction();
        test_set_get();
        test_dense_matrix_conversion();
        test_matrix_vector_multiplication();
        test_row_slice();
        test_exception_handling();
        test_csr_constructor();

        cout << "\n=== ALL TESTS PASSED ===" << endl;
        return 0;
    } catch (const exception& e) {
        cout << "TEST FAILED: " << e.what() << endl;
        return 1;
    }
}
