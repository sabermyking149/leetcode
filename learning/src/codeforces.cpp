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
        int m = v1.size();
        for (i = 0; i < m; i++) {
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
        int m = v.size();
        for (int i = 0; i < m; i++) {
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


void ECR_180_C()
{
    int i, j;
    int n;

    cin >> n;
    vector<int> a(n);
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }

    int left, right, mid;
    long long ways = 0;
    for (i = n - 1; i >= 2; i--) {
        if (a[i] + a[i - 1] + a[i - 2] <= a[n - 1]) {
            break;
        }
        for (j = i - 1; j >= 1; j--) {
            left = 0;
            right = j - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (a[j] + a[mid] > a[i] && a[n - 1] - a[i] - a[j] < a[mid]) { // 还要满足3个数之和大于a[n - 1]
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // cout << i << " " << j << " " << left << " " << j - left << endl;
            ways += j - left;
        }
    }
    cout << ways << endl;
}

// 交互类型题目模版
// 两种query
int q1(int l, int r)
{
    cout << "1 " << l << " " << r << endl;
    cout.flush();
    int res;
    cin >> res;
    return res;
}
int q2(int l, int r)
{
    cout << "2 " << l << " " << r << endl;
    cout.flush();
    int res;
    cin >> res;
    return res;
}
void CR_1059_D() // 即solve函数
{
    int n;
    int a, b;

    cin >> n;

    a = q1(1, n);
    b = q2(1, n);

    int len = b - a;
    int left, right, mid;

    left = 1;
    right = n;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        if (q1(1, mid) == q2(1, mid)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    cout << "! " << left << " " << left + len - 1 << endl;
    cout.flush();
}


void CR_1062_E()
{
    int i, j;
    int n, k, x;
    int t, num;

    cin >> n >> k >> x;

    vector<int> a(n);
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }

    int left, right, mid;

    left = 1;
    right = x;
    sort(a.begin(), a.end());
    while (left <= right) {
        mid = (right - left) / 2 + left;
        t = k;

        if (a[0] >= mid) {
            t-= a[0] - mid + 1;
        }
        for (i = 1; i < n; i++) {
            num = (a[i] - a[i - 1]) / mid;
            if (num > 1) {
                t -= (a[i] - mid) - (a[i - 1] + mid) + 1;
            }
        }
        if (x - a[n - 1] >= mid) {
            t -= x - a[n - 1] - mid + 1;
        }

        if (t <= 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // right 最小距离的最大值
    // cout << right << endl;

    int d = right;
    if (d == 0) {
        for (i = 0; i < k; i++) {
            cout << i << " ";
        }
        cout << endl;
        return;
    }

    int cur, cnt;
    vector<int> pos;

    t = k;
    if (a[0] >= d) {
        cur = 0;
        num = min(t, a[0] - d + 1);
        for (i = 0; i < num; i++) {
            pos.emplace_back(i);
        }
        t -= num;
    }
    for (i = 1; i < n; i++) {
        num = (a[i] - a[i - 1]) / d;
        if (num > 1) {
            cnt = (a[i] - d) - (a[i - 1] + d) + 1;
            cnt = min(t, cnt);
            cur = a[i - 1] + d;
            for (j = 0; j < cnt; j++) {
                pos.emplace_back(cur + j);
            }
            t -= cnt;
            if (t == 0) {
                break;
            }
        }
    }
    if (x - a[n - 1] >= d) {
        cur = a[n - 1] + d;
        for (i = 0; i < t; i++) {
            pos.emplace_back(cur + i);
        }
    }
    for (auto p : pos) cout << p << " ";
    cout << endl;
}


void CR_1062_G()
{
    int i, j;
    int n;

    cin >> n;

    vector<long long> a(n), c(n);
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }
    for (i = 0; i < n; i++) {
        cin >> c[i];
    }

    // 去重
    set<long long> s(a.begin(), a.end());
    vector<long long> v(s.begin(), s.end());

    // 预处理, a[0] 变为v[j]的代价
    int m = v.size();
    vector<long long> dp(m); // dp[j] - 将当前处理元素变为v[j]的最小总代价
    for (j = 0; j < m; j++) {
        if (v[j] == a[0]) {
            dp[j] = 0;
        } else {
            dp[j] = c[0];
        }
    }

    long long prefixMin;
    vector<long long> n_dp;
    for (i = 1; i < n; i++) {
        prefixMin = LLONG_MAX;
        n_dp.assign(m, LLONG_MAX);
        for (j = 0; j < m; j++) {
            if (prefixMin > dp[j]) {
                prefixMin = dp[j];
            }
            if (prefixMin == LLONG_MAX) {
                continue;
            }
            n_dp[j] = prefixMin + (a[i] == v[j] ? 0 : c[i]);
        }
        dp = move(n_dp);
    }
    cout << *min_element(dp.begin(), dp.end()) << endl;
}


// Testing Round 20
// B
void Locate()
{
    int i, k;
    int t, n, x;
    int left, right, mid;
    int l, r;
    int b, res, pos;
    string type;
    vector<int> v;

    cin >> type;
    if (type == "first") {
        cin >> t;
        for (i = 0; i < t; i++) {
            cin >> n;
            v.assign(n + 1, 0);
            b = -1;
            for (k = 1; k <= n; k++) {
                cin >> v[k];
                if (v[k] == 1) {
                    if (b == -1) {
                        x = 1;
                        b = 0;
                    }
                } else if (v[k] == n) {
                    if (b == -1) {
                        x = 0;
                        b = 0;
                    }
                }
            }
            cout << x << endl;
            cout.flush();
        }
        // 输出-1表示结束输入
        cout << -1 << endl;
        cout.flush();
    } else {
        cin >> t;
        for (i = 0; i < t; i++) {
            cin >> n >> x;

            // 右边界
            left = 1;
            right = n;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                cout << "? 1 " << mid << endl;
                cout.flush();
                cin >> res;

                if (res == n - 1) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            r = left;

            // 左边界
            left = 1;
            right = r;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                cout << "? " << mid << " " << r << endl;
                cout.flush();
                cin >> res;

                if (res == n - 1) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            l = right;

            // x 表示大的数的位置
            if (x == 0) {
                pos = l;
            } else {
                pos = r;
            }
            cout << "! " << pos << endl;
            cout.flush();
        }
    }
}


void CR_1065_C2()
{
    int i, j;
    int n;
    int s;
    cin >> n;
    vector<int> a(n), b(n);

    s = 0;
    for (i = 0; i < n; i++) {
        cin >> a[i];
        s ^= a[i];
    }
    for (i = 0; i < n; i++) {
        cin >> b[i];
        s ^= b[i];
    }
    // s最多21位
    for (i = 20; i >= 0; i--) {
        if ((s & (1 << i)) == (1 << i)) {
            // 最后一个控制最高位的一定是胜者
            for (j = n - 1; j >= 0; j--) {
                if (((a[j] & (1 << i)) == (1 << i) && (b[j] & (1 << i)) == 0) || 
                    ((a[j] & (1 << i)) == 0 && (b[j] & (1 << i)) == (1 << i))) {
                    if (j % 2 == 0) {
                        cout << "Ajisai\n";
                    } else {
                        cout << "Mai\n";
                    }
                    return;
                }
            }
        }
    }
    cout << "Tie\n";
}