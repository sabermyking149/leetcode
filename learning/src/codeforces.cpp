#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <stack>
#include <functional>
#include <climits>
#include <iomanip>
using namespace std;

void CR_981_D()
{
    int n, i;
    int ans;
    long long prefix;
    // 由于测试数据原因, 此处用unordered_map反而效率低（CF会出卡哈希的测试数据）
    map<long long, int> sumIdx;

    cin >> n;

    vector<int> v(n), dp(n, 0);
    for (i = 0; i < n; i++) {
        cin >> v[i];
    }
    
    if (v[0] == 0) {
        dp[0] = 1;
    } else {
        dp[0] = 0;
    }
    prefix = v[0];
    ans = dp[0];
    sumIdx[prefix] = 0;
    for (i = 1; i < n; i++) {
        if (v[i] == 0) {
            dp[i] = ans + 1;
        } else {
            prefix += v[i];
            if (sumIdx.count(prefix)) {
                auto idx = sumIdx[prefix];
                dp[i] = max(ans, dp[idx] + 1);
            } else {
                if (prefix == 0) {
                    dp[i] = max(ans, 1);
                } else {
                    dp[i] = max(ans, dp[i - 1]);
                }
            }
        }
        sumIdx[prefix] = i;
        ans = max(ans, dp[i]);
    }
    cout << ans << endl;
}


void CR_991_D()
{
    int i;
    int n;
    string s;

    cin >> s;

    stack<char> st;
    stack<char> st_poped;
    n = s.size();
    // 类似一个单调递减栈, 要保留出栈元素
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(s[i]);
            continue;
        }
        while (s[i] >= '1' && s[i] - 1 > st.top()) {
            st_poped.push(st.top());
            st.pop();
            s[i]--;
            if (st.empty()) {
                break;
            }
        }
        st.push(s[i]);
        while (!st_poped.empty()) {
            st.push(st_poped.top());
            st_poped.pop();
        }
    }
    string ans;
    while (!st.empty()) {
        ans += st.top();
        st.pop();
    }
    reverse(ans.begin(), ans.end());
    cout << ans << endl;
}


void CR_991_E()
{
    int i, j;
    string a, b, c;

    cin >> a >> b >> c;

    int m = a.size();
    int n = b.size();
    // m + n = c.size();
    // dp[i][j] - a前i位与b前j位组成的c[i + j] 最少修改次数
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0x3f3f3f3f));

    // 边界
    dp[0][0] = 0;
    for (i = 0; i < m; i++) {
        if (a[i] != c[i]) {
            dp[i + 1][0] = dp[i][0] + 1;
        } else {
            dp[i + 1][0] = dp[i][0];
        }
    }

    for (i = 0; i < n; i++) {
        if (b[i] != c[i]) {
            dp[0][i + 1] = dp[0][i] + 1;
        } else {
            dp[0][i + 1] = dp[0][i];
        }
    }
    // cout << dp[1][0] << endl;
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (b[j - 1] == c[i + j - 1]) {
                dp[i][j] = min(dp[i][j], dp[i][j - 1]);
            } else {
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1);
            }
 
            if (a[i - 1] == c[i + j - 1]) {
                dp[i][j] = min(dp[i][j], dp[i - 1][j]);
            } else {
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1);
            }
        }
    }
    cout << dp[m][n] << endl;
}


void CR_993_E()
{
    int i;
    long long left, right, mid;
    long long k, l1, r1, l2, r2;
    long long a, b;
    long long ans = 0;

    cin >> k >> l1 >> r1 >> l2 >> r2;

    for (i = 0; i <= 30; i++) {
        if (pow(k, i) > r2) {
            break;
        }
        left = l1;
        right = r1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (mid * pow(k, i) < l2) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        a = left;

        left = l1;
        right = r1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (mid * pow(k, i) > r2) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        b = right;

        if (b >= a) {
            ans += b - a + 1;
        }
    }
    cout << ans << endl;
}


void CR_993_G1()
{
    // 拓扑排序
    int n, r;
    int i;

    cin >> n;
    vector<int> inDegree(n + 1, 0); // 节点入度
    vector<vector<int>> edges(n + 1);
    for (i = 1; i <= n; i++) {
        cin >> r;
        inDegree[r]++;
        edges[i].emplace_back(r);
    }
    queue<int> q;
    for (i = 1; i <= n; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    int cnt = 0;
    while (q.size()) {
        int size = q.size();
        for (i = 0; i < size; i++) {
            auto t = q.front();
            q.pop();
            auto to = edges[t][0];
            inDegree[to]--;
            if (inDegree[to] == 0) {
                q.push(to);
            }
        }
        cnt++;
    }
    cout << cnt + 2 << endl;
}


void CR_995_D()
{
    int i;
    int n;
    long long x, y;
    long long sum, ans, target;
    int leftBound, rightBound;

    cin >> n >> x >> y;
        
    vector<long long> a(n);
    sum = 0;
    for (i = 0; i < n; i++) {
        cin >> a[i];
        sum += a[i];
    }
    sort(a.begin(), a.end());
    ans = 0;
    int left, right, mid;
    for (i = 0; i < n - 1; i++) {
        if (sum - a[i] < x) {
            break;
        }
        // 左边界
        left = i + 1;
        right = n - 1;
        target = sum - a[i] - y;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (a[mid] >= target) {
                right = mid - 1;
                
            } else {
                left = mid + 1;
            }
        }
        leftBound = left;

        // 右边界
        left = i + 1;
        right = n - 1;
        target = sum - a[i] - x;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (a[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        rightBound = right;
        if (rightBound >= leftBound) {
            // printf ("i = %d %d %d\n", i, leftBound, rightBound);
            ans += rightBound - leftBound + 1;
        }
    }
    cout << ans << endl;
}