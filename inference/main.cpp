#include <iostream>
#include <modules/dense.hpp>
using namespace std;

int main(){
    cout << "main working" << endl;

    Dense dense(10);
    dense.printTemp();

    return 0;
}