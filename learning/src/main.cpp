#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>
#include <iterator>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cassert>
#include <string>
#include <numeric>
#include <memory>
#include <thread>
#include <unistd.h>

#include "pub.h"
#include "leetcode.h"
#include "classOper.h"
#include "purec.h"
#include "atcoder.h"



using namespace std;

void xfunc()
{
    static int x = 1;
    cout << "x = " << x << endl;
    x = 2;
    cout << "x = " << x << endl;
}
template <typename T>
void PrintVector(vector<T>& in)
{
    for (auto i : in) {
        cout << i << " ";
    }
    cout << endl;
}

int S(char a) {};
void S() {}
void S(int) {}
void S(double, double=1.2) {}
void S(const char*,const char*) {}


Complex operator-(const Complex& a, const Complex& b)
{
    Complex temp(a.GetReal() - b.GetReal(), a.GetImaginary() - b.GetImaginary());
    return temp;
}

double operator /(const Complex &a, const Complex &b)
{
    return a.real * 1.0 /  (b.real == 0 ? 1 : b.real);
}
vector<int> operator +(vector<int>& a, vector<int>& b)
{
    int i;
    int n = a.size();
    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}

class A {
public:
    int t;
    virtual void fun() {};
private:
    char ch;
    double dou;
};

class B {
    static const int bb;
    static constexpr double bb1 = 1.0;
};

const int B::bb = 1;

class C : public B {
public:
    int aa;
};

class D {
public:
    string _t;
};
class E : virtual public D {
public:
    string t;
};
class F : virtual public D {
public:
    string t;
};

class G : public E, public F {
public:
    string Get()
    {
        return D::_t;
    }
};
class Parent {
public:
    int a;
    Parent()
    {
        cout << "parent construct\n";
    }
    virtual ~Parent()
    {
        cout << "parent destruct\n";
        // system ("pause");
    }
    virtual void func() const {
        cout << "parent func()\n";
    }
};

class Son : public Parent {
public:
    Son()
    {
        cout << "Son construct\n";
    }
    ~Son()
    {
        cout << "Son destruct\n";
        // system ("pause");
    }
    virtual void func() {
        cout << "Son func()\n";
    }
private:
    double b;
};
vector<int> fib_seq(int index)
{
    static vector<int> fibon;

    int i;
    for (i = 0; i <= index; i++) {
        if (i == 0 || i == 1) {
            fibon.emplace_back(1);
        } else {
            fibon.emplace_back(fibon[fibon.size() - 1] + fibon[fibon.size() - 2]);
        }
    }
    return fibon;
}

bool GetSeq(int idx, vector<int> (*seq_f)(int index))
{
    vector<int> seq = seq_f(idx);

    for (auto s : seq) {
        cout << s << " ";
    }
    cout << endl;
    return true;
}

long long taskSchedulerII(vector<int>& tasks, int space)
{
    long long ans = 0;
    int i;
    int n = tasks.size();
    unordered_map<int, long long> um;
    for (i = 0; i < n; i++) {
        if (um.find(tasks[i]) == um.end()) {
            ans++;
            um[tasks[i]] = ans + space + 1;       
        } else if (um[tasks[i]] > ans) {
            ans = um[tasks[i]];
            um[tasks[i]] = ans + space + 1;
            
        } else {      
            ans++;
            um[tasks[i]] = ans + space + 1;
        }
    }
    return ans;
}

vector<int> TransLog(string& log)
{
    vector<int> ans(2, 0);
    int cnt = 0;
    int i, j, a, b;
    int n = log.size();

    j = 0;
    for (i = 0; i < n; i++) {
        if (log[i] == ':') {
            if (cnt == 0) {
                ans[j] = atoi(log.substr(0, i).c_str());
                j++;
                cnt++;
            } else if (cnt == 1) {
                ans[j] = atoi(log.substr(i + 1).c_str());
                j++;
                cnt++;
            }
        }
    }
    return ans;
}
vector<int> exclusiveTime(int n, vector<string>& logs)
{
    // constexpr int stackSize = n;
    // stack<int> s[stackSize];
    int i;
    vector<int> ans;

    for (i = 0; i < n; i++) {
        vector<int> t = TransLog(logs[i]);
        cout << t[0] << " " << t[1] << endl;
    }
    return ans;
}

void PreoderGetGraph(TreeNode *node, TreeNode *parent, unordered_map<int, unordered_set<int>>& edges)
{
    if (node == nullptr) {
        return;
    }
    if (parent != nullptr) {
        edges[node->val].insert(parent->val);
    }
    if (node->left != nullptr) {
        edges[node->val].insert(node->left->val);
    }
    if (node->right != nullptr) {
        edges[node->val].insert(node->right->val);
    }
    PreoderGetGraph(node->left, node, edges);
    PreoderGetGraph(node->right, node, edges);
}
int amountOfTime(TreeNode* root, int start)
{
    int i;
    int n;
    int ans = 0;
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> visited;
    
    PreoderGetGraph(root, nullptr, edges);
    
    queue<int> q;
    q.push(start);
    while (q.size() != 0) {
        n = q.size();
        for (i = 0; i < n; i++) {
            int t = q.front();
            q.pop();
            for (auto it : edges[t]) {
                if (visited.find(it) == visited.end()) {
                    q.push(it);
                }
            }
            visited.insert(t);
        }
        ans++;
    }
    return ans - 1;
}

int scoreOfParentheses(string s)
{
    int i;
    int n = s.size();
    string t;
    stack<char> st;
    char ch;
    // C++ replace没有实现全串替换关键字符串
    st.push(s[0]);
    for (i = 1; i < n; i++) {
        ch = st.top();
        if (s[i] == ')' && ch == '(') {
            st.pop();
            st.push('1');
        } else {
            st.push(s[i]);
        }
    }
    while (st.size()) {
        t += st.top();
        st.pop();
    }
    reverse(t.begin(), t.end());
    // cout << t << endl;
    int ans = 0;
    int left = 0;
    for (i = 0; i < t.size(); i++) {
        if (t[i] == '(') {
            left++;
        } else if (t[i] == ')') {
            left--;
        } else {
            ans += 1 * static_cast<int>(pow(2, left));
        }
    }
    return ans;
}

void CreateLoop(int cur, unordered_map<int, int> relative, bool &findEnd, unordered_set<int>& loopNode)
{
    if (findEnd) {
        return;
    }
    if (loopNode.find(cur) != loopNode.end()) {
        findEnd = true;
        return;
    }
    
    loopNode.emplace(cur);
    CreateLoop(relative[cur], relative, findEnd, loopNode);
}
void DFSFindLoop(unordered_map<int, unordered_set<int>>& e, int cur, int parent, 
    vector<int>& visited, unordered_map<int, int>& relative, bool &findLoop, unordered_set<int>& loopNode)
{
    if (findLoop) {
        return;
    }
    relative[cur] = parent;
    if (visited[cur] == 1) {
        findLoop = true;
        bool findEnd = false;
        CreateLoop(cur, relative, findEnd, loopNode);
        return;
    }
    visited[cur] = 1;
    for (auto it : e[cur]) {
        if (it != parent) {
            DFSFindLoop(e, it, cur, visited, relative, findLoop, loopNode);
        }
    }
    visited[cur] = 2;
}
vector<int> findRedundantConnection(vector<vector<int>>& edges)
{
    int i;
    unordered_map<int, unordered_set<int>> e;
    
    unordered_set<int> nodes;
    for (auto edge : edges) {
        nodes.emplace(edge[0]);
        nodes.emplace(edge[1]);
        e[edge[0]].emplace(edge[1]);
        e[edge[1]].emplace(edge[0]);
    }
    int n = nodes.size() + 1;
    vector<int> visited(n, 0);
    unordered_set<int> loopNode;
    unordered_map<int, int> relative;
    bool findLoop = false;

    DFSFindLoop(e, 1, 0, visited, relative, findLoop, loopNode);
    int index = 0;
    for (i = edges.size() - 1; i >= 0; i--) {
        if (loopNode.find(edges[i][0]) != loopNode.end() && 
            loopNode.find(edges[i][1]) != loopNode.end()) {
                index = i;
                break;
            }
    }
    return edges[index];    
}

unordered_map<int, vector<int>> divisorPos;
unordered_set<int> Divide(int num, int idx)
{
    int i;
    unordered_set<int> divisor;
    
    i = 2;
    while (1) {
        if (num % i == 0) {
            //cout << i << endl;
            divisorPos[i].emplace_back(idx);
            divisor.insert(i);
            while (num % i == 0) {
                num /= i;
            }
        }
        if (num == 1) {
            break;
        }
        i++;
        if (i > sqrt(num)) {
            divisorPos[num].emplace_back(idx);
            divisor.insert(num);
            break;
        }
    }
    return divisor;
}
int findValidSplit(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int idx, curIdx;
    if (n == 1) {
        return -1;
    }
    vector<unordered_set<int>> product(n);
    for (i = 0; i < n; i++) {
        product[i] = Divide(nums[i], i);
    }

    idx = 0;
    for (auto it : product[0]) {
        auto t = divisorPos[it];
        idx = max(idx, t[t.size() - 1]);
    }
    curIdx = idx;
    if (curIdx == 0) {
        return 0;
    }
    for (i = 1; i < n; i++) {
        idx = 0;
        for (auto it : product[i]) {
            auto t = divisorPos[it];
            idx = max(idx, t[t.size() - 1]);
        }
        curIdx = max(curIdx, idx);
        if (curIdx == n - 1) {
            return -1;
        }
        if (curIdx == i) {
            return i;
        }
    }
    return -1;
}

int MySum(int n)
{
    n != 0 && (n += MySum(n - 1));
    return n;
}



void MyQuickSort(vector<int>& arr, int left, int right)
{
    if (left >= right) {
        return;
    }

    int i;
    int n = arr.size();
    int key, size;
    int a, b;

    size = right - left + 1;
    vector<int> t(size, 0);

    key = arr[left];
    a = 0;
    b = size - 1;
    for (i = left; i <= right; i++) {
        if (arr[i] < key) {
            t[a] = arr[i];
            a++;
        } else if (arr[i] > key) {
            t[b] = arr[i];
            b--;
        }
    }
    for (i = a; i <= b; i++) {
        t[i] = key;
    }
    for (i = 0; i < size; i++) {
        arr[i + left] = t[i];
    }

    MyQuickSort(arr, left, left + a - 1);
    MyQuickSort(arr, left + a + 1, right);
}

void Merge(vector<int>& arr, vector<int>& t, int left, int mid, int right)
{
    int i, j;
    int idx;

    i = left;
    j = mid + 1;
    idx = 0;
    while (i <= mid && j <= right) {
        if (arr[i] < arr[j]) {
            t[idx] = arr[i];
            i++;
        } else {
            t[idx] = arr[j];
            j++;
        }
        idx++;
    }
    while (i <= mid) {
        t[idx] = arr[i];
        i++;
        idx++;
    }
    while (j <= right) {
        t[idx] = arr[j];
        j++;
        idx++;
    }
    for (i = left; i <= right; i++) {
        arr[i] = t[i - left];
    }
}
void Divide(vector<int>& arr, vector<int>& t, int left, int right)
{
    int mid;
    if (left < right) {
        mid = (right - left) / 2 + left;
        Divide(arr, t, left, mid);
        Divide(arr, t, mid + 1, right);

        Merge(arr, t, left, mid, right);
    }
}
void MyMergeSort(vector<int>& arr, int left, int right)
{
    int n = arr.size();
    vector<int> t(n, 0);

    Divide(arr, t, left, right);
}

Parent par;
int main0(int argc, char *argv[])
{
#if 0
    cout << "abc\n";
    
    Parent *p1 = new Son; 
    p1->func();

    Parent *p2 = p1;
    p2->func();

    printf ("Hello world![%d]\n", w);
    cout << "When can I be a millionaire?\n";

    vector<string> vv = {"I", "am", "Liwei"};
    PrintVector(vv);

    /*vector<int> fibon = fib_seq(20);
    for (auto f : fibon) {
        cout << f << " ";
    }
    cout << endl;
    */
    GetSeq(20, fib_seq);
    S(2.4);
    S('a');
    delete p1;

    vector<int> tasks = {1,2,1,2,3,1};
    cout << taskSchedulerII(tasks, 3) << endl;
    
    istream_iterator<string> is(cin);
    istream_iterator<string> abc;

    vector<string> text;
    copy(is, abc, back_inserter(text));
    sort(text.begin(), text.end());

    ostream_iterator<string> os(cout, " ");
    copy(text.begin(), text.end(), os);
#endif
    //vector<string> v = {"..E.",".EOW","..W."};
    //ballGame(4, v);
    volatile int i = 10;
    cout << sizeof(Parent) << endl;
    cout << sizeof(Son) << endl;
    cout << sizeof(A) << " " << sizeof(B) << " " << sizeof(C) << endl;

    G gg;
    gg.E::_t = "ab";
    cout << gg.Get() << endl;
    
    gg.F::_t = "cd";
    cout << gg.Get() << endl;
    cout << gg.D::_t << endl;

    Parent *p = new Son();
    p->func();
    cout << sizeof(p) << endl;
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);

    DeleteTree(root);
    assert(root == nullptr);
    // assert(printf ("%d\n", root->val));
    Complex a(5, 2);
    Complex b(3, -8);
    Complex c = a + b;
    Complex d(a);
    Complex e = d - b;
    Complex::Show(c);
    Complex::Show(d);
    Complex::Show(e);
    Complex f(10, 10);
    Complex g = f;
    g = f;
    Complex h = f - g;
    cout << "c / e = " << c / e << endl;

    cout << (a > b) << endl;
    cout << std::gcd(120, 210) << endl;
    cout << func(1) << " " << func(2) << " " << func(3) << endl;
    cout << func(5) << " " << func(8) << " " << func(1000000000000) << endl;

    vector<int> v1 = {1, 2};
    vector<int> v2 = {3, 4};
    vector<int> v3 = v1 + v2;
    cout << v3[0] << " " << v3[1] << endl;
    minOperations(54);

    // OnlyOnHeap ooh1;
    OnlyOnHeap *ooh2 = OnlyOnHeap::create();
    ooh2->destroy();
    OnlyOnStack oos1;
    // OnlyOnStack *oos2 = new OnlyOnStack;

    vector<int> nums = {1,78,27,48,14,86,79,68,77,20,57,21,18,67,5,51,70,85,47,56,22,79,41,8,39,81,59,74,14,45,49,15,10,28,16,77,22,65,8,36,79,94,44,80,72,8,96,78,39,92,69,55,9,44,26,76,40,77,16,69,40,64,12,48,66,7,59,10};
     cout << maxNumOfMarkedIndices(nums) << endl;

    AA aa;
    AA bb;
    cout << &aa << endl;
    cout << &aa.num2 << endl;
    cout << aa.GetArrPoint() << endl;

    cout << &bb << endl;
    cout << &bb.num2 << endl;
    cout << bb.GetArrPoint() << endl;
    cout << endl;

    AA cc;
    AA dd = cc;
    cout << &cc << endl;
    cout << &cc.num2 << endl;
    cout << cc.GetArrPoint() << endl;

    cout << &dd << endl;
    cout << &dd.num2 << endl;
    cout << dd.GetArrPoint() << endl;
    
    unordered_set<int> usa, usb;
    usa.insert(1);
    usa.insert(2);
    usb.insert(2);
    usa.insert(usb.begin(), usb.end());
    for (auto it : usa) {
        cout << it << endl;
    }
    vector<unordered_set<int>> vus(2);
    vus[0] = usa;
    vus[1] = usb;

    cout << add(45, 551) << endl;

    xfunc();
    xfunc();

    vector<string> vs1 = {"60,64: 0.0250","18,43: 0.0169","27,67: 0.0385","2,56: 0.0164","35,59: 0.0161","27,76: 0.0149","19,71: 0.0278","36,87: 0.0137","69,80: 0.0130","44,90: 0.0278","2,49: 0.0133","19,94: 0.0175","3,73: 0.0250","27,93: 0.0135","43,44: 0.0182","3,68: 0.0112","44,78: 0.0149","45,83: 0.0161","20,72: 0.0435","12,82: 0.0164","78,85: 0.0132","12,92: 0.0400","70,80: 0.0167","4,87: 0.0208","53,64: 0.0182","28,75: 0.0208","18,31: 0.0182","9,37: 0.0189","18,28: 0.0244","4,79: 0.0159","2,13: 0.0192","20,88: 0.0185","12,76: 0.0172","45,75: 0.0128","20,84: 0.0263","50,52: 0.0179","53,86: 0.0238","0,63: 0.0147","58,69: 0.0143","25,66: 0.0244","92,98: 0.0185","4,58: 0.0185","4,59: 0.0227","20,38: 0.0185","75,89: 0.0244","25,72: 0.0208","4,44: 0.0208","37,42: 0.0278","58,87: 0.0189","10,99: 0.0147","19,21: 0.0179","50,87: 0.0156","42,76: 0.0192","75,78: 0.0128","19,47: 0.0244","3,56: 0.0133","91,98: 0.0108","51,69: 0.0238","10,91: 0.0159","2,79: 0.0156","20,29: 0.0323","43,70: 0.0182","44,45: 0.0149","12,14: 0.0182","65,94: 0.0172","6,56: 0.0217","40,90: 0.0370","48,86: 0.0714","13,21: 0.0145","13,20: 0.0286","56,67: 0.0167","7,84: 0.0179","14,34: 0.0185","7,75: 0.0164","23,95: 0.0123","30,56: 0.0233","40,66: 0.0455","24,72: 0.0233","16,85: 0.0182","57,66: 0.0208","5,48: 0.0476","13,56: 0.0161","23,98: 0.0102","74,90: 0.0313","46,62: 0.0400","16,92: 0.0357","5,55: 0.0333","81,99: 0.0122","21,36: 0.0109","16,65: 0.0270","6,8: 0.0278","33,74: 0.0244","13,37: 0.0204","82,89: 0.0208","21,50: 0.0120","41,78: 0.0179","57,93: 0.0116","74,75: 0.0185","32,95: 0.0294","21,75: 0.0130","54,75: 0.0123","21,79: 0.0123","41,50: 0.0189","13,88: 0.0139","71,87: 0.0217","7,9: 0.0175","14,99: 0.0238","87,90: 0.0278","5,72: 0.0345","21,88: 0.0114","8,46: 0.0244","62,90: 0.0455","38,67: 0.0143","47,81: 0.0167","16,17: 0.0556","40,59: 0.0294","39,90: 0.0435","48,51: 0.1250","47,85: 0.0167","54,96: 0.0172","22,90: 0.0244","23,59: 0.0145","88,91: 0.0111","23,58: 0.0127","39,77: 0.0435","72,77: 0.0370","55,95: 0.0213","31,56: 0.0159","38,97: 0.0161","63,93: 0.0112","39,71: 0.0303","30,93: 0.0189"};
    vector<string> vs2 = {"0,63: 0.0147","2,13: 0.0192","2,49: 0.0133","2,56: 0.0164","2,79: 0.0156","3,56: 0.0133","3,68: 0.0112","3,73: 0.0250","4,44: 0.0208","4,58: 0.0185","4,59: 0.0227","4,79: 0.0159","4,87: 0.0208","5,48: 0.0476","5,55: 0.0333","5,72: 0.0345","6,8: 0.0278","6,56: 0.0217","7,9: 0.0175","7,75: 0.0164","7,84: 0.0179","8,46: 0.0244","9,37: 0.0189","10,91: 0.0159","10,99: 0.0147","12,14: 0.0182","12,76: 0.0172","12,82: 0.0164","12,92: 0.0400","13,20: 0.0286","13,21: 0.0145","13,37: 0.0204","13,56: 0.0161","13,88: 0.0139","14,34: 0.0185","14,99: 0.0238","16,17: 0.0556","16,65: 0.0270","16,85: 0.0182","16,92: 0.0357","18,28: 0.0244","18,31: 0.0182","18,43: 0.0169","19,21: 0.0179","19,47: 0.0244","19,71: 0.0278","19,94: 0.0175","20,29: 0.0323","20,38: 0.0185","20,72: 0.0435","20,84: 0.0263","20,88: 0.0185","21,36: 0.0109","21,50: 0.0120","21,75: 0.0130","21,79: 0.0123","21,88: 0.0114","22,90: 0.0244","23,58: 0.0127","23,59: 0.0145","23,95: 0.0123","23,98: 0.0102","24,72: 0.0233","25,66: 0.0244","25,72: 0.0208","27,67: 0.0385","27,76: 0.0149","27,93: 0.0135","28,75: 0.0208","30,56: 0.0233","30,93: 0.0189","31,56: 0.0159","32,95: 0.0294","33,74: 0.0244","35,59: 0.0161","36,87: 0.0137","37,42: 0.0278","38,67: 0.0143","38,97: 0.0161","39,71: 0.0303","39,77: 0.0435","39,90: 0.0435","40,59: 0.0294","40,66: 0.0455","40,90: 0.0370","41,50: 0.0189","41,78: 0.0179","42,76: 0.0192","43,44: 0.0182","43,70: 0.0182","44,45: 0.0149","44,78: 0.0149","44,90: 0.0278","45,75: 0.0128","45,83: 0.0161","46,62: 0.0400","47,81: 0.0167","47,85: 0.0167","48,51: 0.1250","48,86: 0.0714","50,52: 0.0179","50,87: 0.0156","51,69: 0.0238","53,64: 0.0182","53,86: 0.0238","54,75: 0.0123","54,96: 0.0172","55,95: 0.0213","56,67: 0.0167","57,66: 0.0208","57,93: 0.0116","58,69: 0.0143","58,87: 0.0189","60,64: 0.0250","62,90: 0.0455","63,93: 0.0112","65,94: 0.0172","69,80: 0.0130","70,80: 0.0167","71,87: 0.0217","72,77: 0.0370","74,75: 0.0185","74,90: 0.0312","75,78: 0.0128","75,89: 0.0244","78,85: 0.0132","81,99: 0.0122","82,89: 0.0208","87,90: 0.0278","88,91: 0.0111","91,98: 0.0108","92,98: 0.0185"
};
    cout << vs1.size() << " " << vs2.size() << endl;
    unordered_set<string> us1, us2;
    for (auto v : vs1) {
        us1.emplace(v);
    }
    for (auto v : vs2) {
        us2.emplace(v);
    }
    unordered_set<string>::iterator it;
    for (it = us1.begin(); it != us1.end();) {
        if (us2.count(*it) == 1) {
            us2.erase(*it);
            us1.erase(it++);
        } else {
            it++;
        }
    }
    for (auto v : us1) cout << v << "  "; cout << endl;
    cout << endl;
    for (auto v : us2) cout << v << "  "; cout << endl;
    cout << endl;
    cout << us1.size() << " " << us2.size() << endl;
    
    int pp = 0x12345678;
    char *charp = (char *)malloc(4);
    charp = (char *)&pp;
    printf ("%d\n", charp[0]);
    cout << MAX(4, 3) << endl;

    vector<string> chessboard = {"X....", "O.X..", "OOOOX", "....."};
    cout << flipChess(chessboard) << endl;

    string instructions = "LLGRL";
    cout << isRobotBounded(instructions) << endl;
    cout << MySum(10) << endl;
    cout << "end of main\n";
    return 0;
}

void PrintString(string& msg)
{
    cout << msg << endl;
}
auto complex(std::make_unique<Complex>(2.1, 4.2));
#if 0
int main(int argc, char *argv[])
{
    unique_ptr<Complex> complex1;
    complex1 = move(complex);
    cout << complex1->aConst << endl;
    Complex::Show(complex1);

    Complex c1(1, 3);
    Complex c2 = c1;
    cout << c2.aConst << endl;

    c1 = c2;
    cout << c1.aConst << endl;

    auto ptr = new int(8);
    auto ptrs = new int[8];
    cout << *ptr << endl;
    cout << *ptrs << endl;

    auto deletor = [] (Complex *pComlex) { delete pComlex; cout << "smart pointer destruct\n"; };
    unique_ptr<Complex, decltype(deletor)> aComplex(new Complex(5, -1.5), deletor);
    cout << aComplex->GetImaginary() << endl;
    vector<int> arr = {5, 3, 7, 6, 4, 1, 0, 2, 9, 10, 8, 5, 5};
    // MyQuickSort(arr, 0, 12);
    MyMergeSort(arr, 0, 12);
    for (auto a : arr) cout << a << " ";
    cout << endl;

    string msg = "hello world";
    //thread th1(PrintString, msg);
    //th1.detach();
    sleep(1);
    return 0;
}
#endif

void pointer()
{
    char *password = "e1234";
    printf ("%p\n", password);
}
int main(int argc, char *argv[])
{
    cout << MySqrt(5.0) << endl;
    cout << MySqrt(0.64) << endl;

    ABC_356_C();
}