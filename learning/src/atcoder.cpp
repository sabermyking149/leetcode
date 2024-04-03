#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

void ABC_346_D()
{
    int n, i, cost;
    string s;

    cin >> n >> s;
    vector<int> costs(n);
    for (i = 0; i < n; i++) {
        cin >> cost;
        costs[i] = cost;
    }
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(2, vector<long long>(2, LLONG_MAX)));
    // dp[i][0][0] - 第i位不改变且当前连续字符字段个数为0的最小花费
    dp[0][0][0] = 0;
    dp[0][1][0] = costs[0];
    for (i = 1; i < n; i++) {
        if (s[i] == s[i - 1]) {
            dp[i][0][0] = dp[i - 1][1][0];
            dp[i][0][1] = min(dp[i - 1][0][0], dp[i - 1][1][1]);
            dp[i][1][0] = dp[i - 1][0][0] + costs[i];
            dp[i][1][1] = min(dp[i - 1][1][0], dp[i - 1][0][1]) + costs[i];
        } else {
            dp[i][0][0] = dp[i - 1][0][0];
            dp[i][0][1] = min(dp[i - 1][1][0], dp[i - 1][0][1]);
            dp[i][1][0] = dp[i - 1][1][0] + costs[i];
            dp[i][1][1] = min(dp[i - 1][0][0], dp[i - 1][1][1]) + costs[i];
        }
    }
    long long ans = min(dp[n - 1][0][1], dp[n - 1][1][1]);
    cout << ans << endl;
}


void ARC_175_B()
{
    int i;
    long long n, a, b;
    long long ans;
    string s;

    cin >> n >> a >> b;
    cin >> s;

    int left, right;
    int len = s.size();

    left = right = 0;
    for (i = 0; i < len; i++) {
        if (s[i] == '(') {
            left++;
        } else {
            if (left > 0) {
                left--;
            } else {
                right++;
            }
        }
    }
    long long small = min(right, left);
    long long big = max(right, left);
    long long swap = (small + 1) / 2;
    long long ts, tb, cost;

    ans = LLONG_MAX;
    for (i = 0; i <= swap; i++) {
        if (small - i * 2 >= 0) {
            ts = small - i * 2;
            tb = big - i * 2;
        } else {
            ts = 0;
            tb = big - small;
        }
        cost = i * a + ((ts + 1) / 2 + (tb + 1) / 2) * b;
        ans = min(ans, cost);
    }
    cout << ans << endl;
}


void ABC_347_C()
{
    int i;
    int n;
    int a, b;

    cin >> n >> a >> b;

    vector<int> d(n);
    int weekLen = a + b;
    for (i = 0; i < n; i++) {
        cin >> d[i];
        d[i] %= weekLen;
    }
    sort(d.begin(), d.end());
    if (d[n - 1] - d[0] + 1 <= a) {
        cout << "Yes" << endl;
        return;
    }
    // 工作日在中间
    // 2 5 5
    // 1 7
    for (i = 1; i < n; i++) {
        if (d[i] - d[i - 1] > b) {
            cout << "Yes" << endl;
            return;
        }
    }
    cout << "No" << endl;
}


void ABC_347_D()
{
    int a, b;
    int i, j, k;
    int cnt0, cnt1;
    unsigned long long c, n;

    cin >> a >> b >> c;

    n = c;
    cnt0 = cnt1 = 0;
    while (n) {
        if (n % 2 == 1) {
            cnt1++;
        } else {
            cnt0++;
        }
        n >>= 1;
    }

    int same = (a + b - cnt1) / 2;
    int distinctA = a - same;
    int distinctB = b - same;

    // cout << same << distinctA << endl;
    if (a + b < cnt1 || abs(a - b) > cnt1) {
        cout << "-1" << endl;
        return;
    }
    // construct
    unsigned long long A, B;
    bool f;

    j = i = 0;
    A = B = 0;
    if (same >= cnt0) {
        f = true;
    } else {
        k = cnt0 - same;
        f = false;
    }
    while (1) {
        if (distinctA == 0 && distinctB == 0 && same == 0) {
            break;
        }
        if ((c & 1ull << i) == 1ull << i) {
            if (distinctA > 0) {
                A += 1ull << i;
                distinctA--;
            } else {
                B += 1ull << i;
                distinctB--;
            }
        } else {
            if (same > 0) {
                if (f) {
                    A += 1ull << i;
                    B += 1ull << i;
                    same--;
                } else {
                    j++;
                    if (j > k) {
                        A += 1ull << i;
                        B += 1ull << i;
                        same--;
                    }
                }
            }
        }
        i++;
        // 超过ull范围
        if (i > 63) {
            cout << "-1" << endl;
            return;
        }
    }
    cout << A << " " << B << endl;
}