#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
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
#include "pub.h"

using namespace std;

void Monthly_Round_136_E()
{
    int n, m;
    int inf = 0x3f3f3f3f;
    string s;
    cin >> n >> m >> s;

    int i, j;
    vector<string> grid(n);
    for (i = 0; i < n; i++) {
        cin >> grid[i];
    }

    // 反向bfs, 3进制状态压缩
    // dist[i][j] - 从pos i 到达终点(m - 1, n - 1) 且pos处的字符串的三进制编码是j的最少步骤
    vector<vector<int>> dist(m * n, vector<int>(27, inf));
    queue<tuple<int, int, int>> q; // 距离 - 位置 - 三进制编码
    // 终点所有可能的三进制编码都可以
    for (j = 0; j < 27; j++) {
        dist[m * n - 1][j] = 0;
        q.push({0, m * n - 1, j});
    }

    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    vector<int> b(3);
    while (!q.empty()) {
        auto [d, pos, mask] = q.front();
        q.pop();
        if (dist[pos][mask] < d) {
            continue;
        }

        auto r = pos / m;
        auto c = pos % m;
        for (i = 0; i < 4; i++) {
            for (j = 1; j <= 3; j++) {
                auto nr = r + directions[i][0] * j;
                auto nc = c + directions[i][1] * j;

                if (nr < 0 || nr >= n || nc < 0 || nc >= m) {
                    continue;
                }
                auto npos = nr * m + nc;
                // 反向求n_mask: b[0] b[1] b[2] -> b[2] b[0] b[1]
                b[0] = mask % 3;
                b[1] = mask / 3 % 3;
                b[2] = mask / 9;
                auto n_mask = b[2] + b[0] * 3 + b[1] * 9;

                // n_mask -> mask 是否与移动步数矛盾
                char ch;
                if (j == 1) {
                    ch = n_mask % 3 + 'a';
                } else if (j == 2) {
                    ch = n_mask / 3 % 3 + 'a';
                } else {
                    ch = n_mask / 9 + 'a';
                }
                if (ch == grid[r][c] && dist[npos][n_mask] > d + 1) {
                    dist[npos][n_mask] = d + 1;
                    q.push({d + 1, npos, n_mask});
                }
            }
        }
    }
    int curMask = (s[0] - 'a') + (s[1] - 'a') * 3 + (s[2] - 'a') * 9;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (dist[i * m + j][curMask] == inf) {
                cout << -1 << " ";
            } else {
                cout << dist[i * m + j][curMask] << " ";
            }
        }
        cout << "\n";
    }
}