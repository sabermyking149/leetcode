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