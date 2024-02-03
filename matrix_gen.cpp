#include <bits/stdc++.h>
#include <fstream>
#include "constants.h"
using namespace std;

int main(){
    ofstream fout;
    fout.open(INPUT_MATRIX_FILE);
    vector<vector<double>> A(N, vector<double>(N, 0.0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i][j] = drand48();
            fout << A[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
    return 0;
}