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

#include "pub.h"
#include "leetcode.h"
#include "classOper.h"


using namespace std;

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
private:
	double real;
	double imaginary;
};

Complex operator-(const Complex& a, const Complex& b)
{
	Complex temp(a.GetReal() - b.GetReal(), a.GetImaginary() - b.GetImaginary());
	return temp;
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

};

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
	int size = logs.size();
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
int main(int argc, char *argv[])
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
	return 0;
}