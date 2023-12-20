#include <modules/dense.hpp>
#include <iostream>
using namespace std;

Dense::Dense(int temp){ //Dense::Dense -> Dense constructor belongs to Dense class
    this->temp = temp;
}

void Dense::printTemp(){
    cout << this->temp << endl;
}