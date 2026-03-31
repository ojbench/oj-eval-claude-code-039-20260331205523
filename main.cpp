#include "CSRMatrix.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace sjtu;

int main() {
    // Test basic functionality
    CSRMatrix<int> mat(3, 4);

    cout << "Rows: " << mat.getRowSize() << endl;
    cout << "Cols: " << mat.getColSize() << endl;
    cout << "NNZ: " << mat.getNonZeroCount() << endl;

    // Test set
    mat.set(0, 1, 5);
    mat.set(1, 2, 3);
    mat.set(2, 0, 7);

    cout << "After set, NNZ: " << mat.getNonZeroCount() << endl;

    // Test get
    cout << "mat(0,1) = " << mat.get(0, 1) << endl;
    cout << "mat(1,2) = " << mat.get(1, 2) << endl;
    cout << "mat(2,0) = " << mat.get(2, 0) << endl;
    cout << "mat(0,0) = " << mat.get(0, 0) << endl;

    // Test getMatrix
    auto dense = mat.getMatrix();
    cout << "Dense matrix:" << endl;
    for (size_t i = 0; i < dense.size(); ++i) {
        for (size_t j = 0; j < dense[i].size(); ++j) {
            cout << dense[i][j] << " ";
        }
        cout << endl;
    }

    // Test matrix-vector multiplication
    vector<int> vec = {1, 2, 3, 4};
    try {
        auto result = mat * vec;
        cout << "Matrix-vector product: ";
        for (auto val : result) {
            cout << val << " ";
        }
        cout << endl;
    } catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }

    // Test row slice
    try {
        auto slice = mat.getRowSlice(0, 2);
        cout << "Row slice [0,2): rows=" << slice.getRowSize()
             << " cols=" << slice.getColSize()
             << " nnz=" << slice.getNonZeroCount() << endl;
    } catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }

    // Test constructor from dense matrix
    vector<vector<int>> dense_input = {
        {0, 1, 0, 2},
        {3, 0, 0, 0},
        {0, 0, 4, 5}
    };
    CSRMatrix<int> mat2(3, 4, dense_input);
    cout << "From dense constructor, NNZ: " << mat2.getNonZeroCount() << endl;

    return 0;
}
