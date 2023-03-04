#ifndef _CLASSOPER_H_
#define _CLASSOPER_H_

#include <iostream>
#include <vector>

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

#endif