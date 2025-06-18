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
#include <numeric>
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


// 找规律
void CR_1016_D()
{
    auto CalcVal = [](int x, int y, int n) {
        int i;
        int direction[4][2] = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
        vector<int> v1, v2;
        long long t = (1ll << (n - 1));
        long long d;
        while (n - 1) {
            for (i = 3; i >= 0; i--) {
                if (x - i * t > 0) {
                    x -= i * t;
                    v1.emplace_back(i);
                    break;
                }
            }
            for (i = 3; i >= 0; i--) {
                if (y - i * t > 0) {
                    y -= i * t;
                    v2.emplace_back(i);
                    break;
                }
            }
            n--;
            t = (1ll << (n - 1));
        }
        reverse(v1.begin(), v1.end());
        reverse(v2.begin(), v2.end());
        if (x == 1 && y == 1) {
            d = 1;
        } else if (x == 2 && y == 2) {
            d = 2;
        } else if (x == 2 && y == 1) {
            d = 3;
        } else {
            d = 4;
        }
        for (i = 0; i < v1.size(); i++) {
            for (int j = 0; j < 4; j++) {
                if (v1[i] == direction[j][0] && v2[i] == direction[j][1]) {
                    d += j * (1ll << (i + 1)) * (1ll << (i + 1));
                    break;
                }
            }
        }
        cout << d << endl;
    };
    auto CalcPos = [](long long d, int n) {
        long long t = (1ll << (n - 1)) * (1ll << (n - 1));
        vector<int> v;
        int x, y;
        while (n - 1) {
            for (int i = 3; i >= 0; i--) {
                if (d - i * t > 0) {
                    d -= i * t;
                    v.emplace_back(i);
                    break;
                }
            }
            n--;
            t = (1ll << (n - 1)) * (1ll << (n - 1));
        }
        // for (auto a : v) cout << a << " ";
        reverse(v.begin(), v.end());
        if (d == 1) {
            x = 1;
            y = 1;
        } else if (d == 2) {
            x = 2;
            y = 2;
        } else if (d == 3) {
            x = 2;
            y = 1;
        } else {
            x = 1;
            y = 2;
        }
        int direction[4][2] = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
        for (int i = 0; i < v.size(); i++) {
            x += direction[v[i]][0] * (1 << (i + 1));
            y += direction[v[i]][1] * (1 << (i + 1));
        }
        cout << x << " " << y << endl;
    };

    int i;
    int n, q;
    long long x, y, d;
    cin >> n >> q;
    string type;
    for (i = 0; i < q; i++) {
        cin >> type;
        if (type == "->") {
            cin >> x >> y;
            CalcVal(x, y, n);
        } else {
            cin >> d;
            CalcPos(d, n);
        }
    };
}


void CR_1017_E()
{
    int i;
    int n, cnt;

    cin >> n;

    long long t;
    vector<long long> a(n);
    vector<vector<int>> bits(31, vector<int>(2, 0));
    for (i = 0; i < n; i++) {
        cin >> a[i];
        t = a[i];
        cnt = 0;
        while (cnt < 31) {
            bits[cnt][t % 2]++;
            t >>= 1;
            cnt++;
        }
    }

    long long contribute, maxVal;
    long long ak;
    maxVal = -1;
    for (i = 0; i < n; i++) {
        t = a[i];
        contribute = 0;
        cnt = 0;
        while (cnt < 31) {
            if (t % 2 == 0) {
                contribute += bits[cnt][1] * (1ll << cnt);
            } else {
                contribute += bits[cnt][0] * (1ll << cnt);
            }
            t >>= 1;
            cnt++;
        }
        // cout << contribute << " ";
        if (contribute > maxVal) {
            ak = a[i];
            maxVal = contribute;
        }
    }

    // cout << "ak = " << ak << endl;
    // 此处maxVal其实就是最终结果
    long long sum = 0;
    for (i = 0; i < n; i++) {
        sum += (ak ^ a[i]);
    }
    cout << sum << endl;
}


// CSP-J 2024 第3题
void P11229()
{
    int n;

    cin >> n;

    // 前13个数字的拼接木棍数
    vector<string> nums = {"-1", "-1", "1", "7", "4", "2", "6", "8", "10", 
        "18", "22", "20", "28", "68"};

    if (n <= 13) {
        cout << nums[n] << endl;
        return;
    }

    int m, r;

    m = n / 7;
    r = n % 7;
    if (r == 1) {
        m--;
        r = 8;
    }

    string a, b;

    a = nums[r] + string(m, '8');

    if (m > 0) {
        if (r + 7 <= 13) {
            if (r + 7 == 10 && m > 1) { // 11根木棍反而能拼出更小的数字
                b += "200";
                m -= 2;
            } else {
                b += nums[r + 7];
                m--;
            }
            b += string(m, '8');
            if (a.size() == b.size()) {
                a = min(a, b);
            } else {
                a = (a.size() < b.size() ? a : b);
            }
        }
    }
    cout << a << endl;
}


void CR_1029_C()
{
    int i;
    int n;

    cin >> n;

    int cnt;
    vector<int> a(n);
    unordered_set<int> prev, cur, t;
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }

    cnt = 1;
    prev.emplace(a[0]);
    for (i = 1; i < n; i++) {
        cur.emplace(a[i]);
        if (prev.count(a[i])) {
            t.emplace(a[i]);
            if (t.size() == prev.size()) {
                cnt++;
                prev = cur;
                t.clear();
            }
        }
    }
    cout << cnt << endl;
}