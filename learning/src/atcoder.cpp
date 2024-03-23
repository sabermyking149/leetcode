#include <iostream>
#include <string>
#include <vector>
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