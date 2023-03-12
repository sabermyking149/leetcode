#ifndef _CLASSOPER_H_
#define _CLASSOPER_H_

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 类只建立在堆上
class OnlyOnHeap {
protected:
    OnlyOnHeap() {}
    ~OnlyOnHeap() {}
public:
    static OnlyOnHeap *create()
    {
        cout << "Create class OnlyOnHeap\n";
        return new OnlyOnHeap();
    }
    void destroy()
    {
        cout << "Destroy class OnlyOnHeap\n";
        delete this;
    }
};

// 类只建立在栈上
class OnlyOnStack {
private:
    void *operator new(size_t t) {}    // 注意函数的第一个参数和返回值都是固定的
    void operator delete(void *ptr) {} // 重载了 new 就需要重载 delete
public:
    OnlyOnStack() {}
    ~OnlyOnStack() {}
};


// 指针计数
class AA {
public:
    double num;
    int num2;
    AA(size_t len = 1)
    {
        intPtr = new int;
        *intPtr = 1;
        arr = new int[len];
    }
    ~AA()
    {
        (*intPtr)--;
        if (*intPtr == 0) {
            delete []arr;
            arr = nullptr;

            delete intPtr;
            intPtr = nullptr;
        }
    }
    AA(const AA &t)
    {
        intPtr = t.intPtr;
        arr = t.arr;
        (*intPtr)++;
    }
    int *GetArrPoint()
    {
        return arr;
    }
private:
    int *arr;
    int *intPtr;
};


class Complex {
public:
    Complex(double r, double i) : real(r), imaginary(i) {}
    Complex(const Complex &c)
    {
        cout << "拷贝构造函数\n";
        real = c.real;
        imaginary = c.imaginary;
    }
    Complex operator +(const Complex &a)
    {
        Complex temp(0, 0);
        temp.real = a.real + real;
        temp.imaginary = a.imaginary + imaginary;
        return temp;
    }
    bool operator >(const Complex &a) {
        return real * real + imaginary * imaginary > a.real * a.real + a.imaginary * a.imaginary;
    }
    Complex &operator =(const Complex &c)
    {
        cout << "重载赋值\n";
        if (&c != this) {
            real = c.real;
            imaginary = c.imaginary;
        }
        return *this;
    }
    static void Show(Complex &c)
    {
        cout << to_string(c.real) + ' ' + (c.imaginary < 0 ? '-' : '+') + ' ' + to_string(fabs(c.imaginary)) + 'i' << endl;
    }
    double GetReal() const
    {
        return real;
    }
    double GetImaginary() const
    {
        return imaginary;
    }
    // friend Complex operator -(const Complex &a, const Complex &b);
private:
    double real;
    double imaginary;
};

#endif