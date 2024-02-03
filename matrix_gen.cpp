#include <bits/stdc++.h>
#include <fstream>
#include "constants.h"
#include <random>
using namespace std;

int main(){

    ofstream fout;
    fout.open(INPUT_MATRIX_FILE);
    if(!fout.is_open()){
        perror("Error opening file");
        return 0;
    }

    vector<vector<double>> A(N, vector<double>(N, 0.0));
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i][j] = distribution(generator);
            fout << A[i][j] << " ";
        }
        fout << endl;
    }
    
    fout.close();
    return 0;
}