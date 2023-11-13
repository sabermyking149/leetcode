#ifndef _LEETCODE_H_
#define _LEETCODE_H_

#include <iostream>
#include <vector>

#include "pub.h"

using namespace std;

int gcd(int a, int b);
void DeleteTree(TreeNode* &node);
int waysToMakeFair(vector<int>& nums);
int func(long long n);
int minOperations(int n);
int maxNumOfMarkedIndices(vector<int>& nums);
int flipChess(vector<string>& chessboard);
bool isRobotBounded(string instructions);
double MySqrt(double num);
#endif