#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <functional>
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


void ABC_348_D()
{
    int i, j;
    int h, w;
    int n;
    int r, c, p;
    int S, T;

    unordered_map<int, int> energy;
    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    cin >> h >> w;

    vector<vector<char>> grid(h, vector<char>(w, '.'));
    vector<vector<int>> cur_energy(h, vector<int>(w, -1));
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            cin >> grid[i][j];
            if (grid[i][j] == 'S') {
                S = i * w + j;
            }
            if (grid[i][j] == 'T') {
                T = i * w + j;
            }
        }
    }
    cin >> n;
    // 二维转一维
    for (i = 0; i < n; i++) {
        cin >> r >> c >> p;
        energy[(r - 1) * w + c - 1] = p;
    }
    if (energy.count(S) == 0) {
        cout << "No" << endl;
        return;
    }
    queue<tuple<int, int, unordered_map<int, int>&>> q;

    q.push({S, 0, energy});
    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        // printf ("{%d %d} = %d\n", front.first / w, front.first % w, front.second);
        if (get<0>(cur) == T) {
            cout << "Yes" << endl;
            return;
        }
        if (cur_energy[get<0>(cur) / w][get<0>(cur) % w] > get<1>(cur)) {
            continue;
        }
        if (get<2>(cur).count(get<0>(cur)) == 1) {
            if (get<1>(cur) >= get<2>(cur)[get<0>(cur)]) {
                cur_energy[get<0>(cur) / w][get<0>(cur) % w] = get<1>(cur);
            } else {
                cur_energy[get<0>(cur) / w][get<0>(cur) % w] = get<2>(cur)[get<0>(cur)];
                get<2>(cur).erase(get<0>(cur));
            }
        } else {
            cur_energy[get<0>(cur) / w][get<0>(cur) % w] = get<1>(cur);
        }
        if (cur_energy[get<0>(cur) / w][get<0>(cur) % w] == 0) {
            continue;
        }
        for (i = 0; i < 4; i++) {
            auto nh = get<0>(cur) / w + directions[i][0];
            auto nw = get<0>(cur) % w + directions[i][1];
            if (nh < 0 || nh >= h || nw < 0 || nw >= w || grid[nh][nw] == '#') {
                continue;
            }
            if (cur_energy[get<0>(cur) / w][get<0>(cur) % w] - 1 > cur_energy[nh][nw]) {
                cur_energy[nh][nw] = cur_energy[get<0>(cur) / w][get<0>(cur) % w] - 1;
                q.push({nh * w + nw, cur_energy[nh][nw], get<2>(cur)});
            }
        }
    }
    cout << "No" << endl;
}


void ABC_350_D()
{
    int i;
    int N, M;
    int n1, n2;
    int cnt, e;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, int> um;

    cin >> N >> M;
    for (i = 0; i < M; i++) {
        cin >> n1 >> n2;
        edges[n1].emplace(n2);
        edges[n2].emplace(n1);
        um[n1]++;
    }

    vector<bool> visited(N + 1, false);
    function<void (unordered_map<int, unordered_set<int>>&, int, int, int&, int&)> dfs = 
        [&dfs, &visited, &um](unordered_map<int, unordered_set<int>>& edges, int cur, int parent, int& cnt, int& e) {
        if (visited[cur]) {
            return;
        }
        visited[cur] = true;
        cnt++;
        e += um[cur];
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                if (it != parent) {
                    dfs(edges, it, cur, cnt, e);
                }
            }
        }
    };
    long long ans = 0;
    for (i = 1; i <= N; i++) {
        if (visited[i] == false) {
            e = cnt = 0;
            dfs(edges, i, -1, cnt, e);
            ans += (long long)(cnt - 1) * cnt / 2 - e;
        }
    }
    cout << ans << endl;
    return;
}


// 接近超时
void ABC_351_D()
{
    int i, j;
    int H, W;
    unordered_set<int> dangerZone;
    cin >> H >> W;

    vector<vector<char>> grid(H, vector<char>(W, ' '));
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            cin >> grid[i][j];
        }
    }
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            if (grid[i][j] == '#') {
                // 相邻四格都为dangerZone;
                if (i - 1 >= 0 && grid[i - 1][j] == '.') {
                    dangerZone.emplace((i - 1) * W + j);
                }
                if (i + 1 < H && grid[i + 1][j] == '.') {
                    dangerZone.emplace((i + 1) * W + j);
                }
                if (j - 1 >= 0 && grid[i][j - 1] == '.') {
                    dangerZone.emplace(i * W + j - 1);
                }
                if (j + 1 < W && grid[i][j + 1] == '.') {
                    dangerZone.emplace(i * W + j + 1);
                }
            }
        }
    }

    vector<vector<bool>> visited(H, vector<bool>(W, false));
    unordered_set<int> visitedDangerZone;

    function<void (vector<vector<char>>&, int, int, int&)> dfs = 
        [&dfs, &visited, &visitedDangerZone, &dangerZone](vector<vector<char>>& grid, int row, int col, int& cnt) {

        int i;
        int pos;
        int H, W;
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        H = grid.size();
        W = grid[0].size();
        pos = row * W + col;
        if (dangerZone.count(pos)) {
            if (visitedDangerZone.count(pos) == 0) {
                visitedDangerZone.emplace(pos);
                cnt++;
            }
            return;
        }
        visited[row][col] = true;
        cnt++;
        for (i = 0; i < 4; i++) {
            auto nr = row + directions[i][0];
            auto nc = col + directions[i][1];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W || grid[nr][nc] != '.' || visited[nr][nc]) {
                continue;
            }
            dfs(grid, nr, nc, cnt);
        }
    };
    int cnt;
    int ans = 1;
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            if (visited[i][j] == false && grid[i][j] == '.' && 
                dangerZone.count(i * W + j) == 0) {
                cnt = 0;
                visitedDangerZone.clear();
                dfs(grid, i, j, cnt);
                ans = max(ans, cnt);
                //cout << cnt << endl;
            }
        }
    }
    cout << ans << endl;
    return;
}


void ABC_353_D()
{
    int i;
    int n;
    int mod = 998244353;
    unordered_map<int, int> bits;
    cin >> n;

    vector<int> v(n);
    for (i = 0; i < n; i++) {
        cin >> v[i];
        bits[v[i]] = to_string(v[i]).size();
    }

    vector<unsigned long long> prefixSum(n, 0);
    vector<unsigned long long> prefixBitSum(n, 0);

    for (i = n - 1; i >= 1; i--) {
        if (i == n - 1) {
            prefixSum[i] = v[i];
            prefixBitSum[i] = pow(10, bits[v[i]]);
        } else {
            prefixSum[i] = (v[i] + prefixSum[i + 1]) % mod;
            prefixBitSum[i] = 
                (static_cast<unsigned long long>(pow(10, bits[v[i]])) + prefixBitSum[i + 1]) % mod;
        }
    }

    unsigned long long ans = 0;
    for (i = 0; i < n - 1; i++) {
        ans = (ans + prefixSum[i + 1] + (prefixBitSum[i + 1] * v[i]) % mod) % mod;
    }
    cout << ans << endl;
    return;
}


// 搜索迷宫可以n次穿墙的可行方法,但效率低
void ARC_177_C()
{
    int i, j;
    int n;

    cin >> n;

    vector<vector<char>> grid(n, vector<char>(n, ' '));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            cin >> grid[i][j];
        }
    }
    char sign;
    int endRow, endCol;

    function<void (vector<vector<vector<char>>>&, int, int, int, int, bool&)> dfs = 
        [&dfs, &sign, &endRow, &endCol](vector<vector<vector<char>>>& allgrid, int row, int col, 
        int canChange, int cnt, bool& canReach) {

        if (canReach) {
            return;
        }
        if (row == endRow && col == endCol) {
            canReach = true;
            return;
        }

        int i;
        int n = allgrid[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        allgrid[cnt][row][col] = 'A';
        for (i = 0; i < 4; i++) {
            auto nr = row + directions[i][0];
            auto nc = col + directions[i][1];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                continue;
            }
            if (allgrid[cnt][nr][nc] == sign) {
                dfs(allgrid, nr, nc, canChange, cnt, canReach);
            } else {
                if (canChange > 0) {
                    // allgrid[cnt + 1] = allgrid[cnt];
                    dfs(allgrid, nr, nc, canChange - 1, cnt + 1, canReach);
                }
            }
        }
    };

    int left, right, mid;
    int cnt1, cnt2;
    bool canReach = false;
    vector<vector<vector<char>>> allgrid; // allgrid[k] - 打破k面墙的grid图

    for (i = 0; i < n * 2; i++) {
        allgrid.emplace_back(grid);
    }

    left = 0;
    right = n * 2 - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        sign = 'R';
        endRow = endCol = n - 1;
        canReach = false;

        auto t = allgrid;
        dfs(t, 0, 0, mid, 0, canReach);
        if (canReach) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    cnt1 = left;

    left = 0;
    right = n * 2 - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        sign = 'B';
        endRow = 0;
        endCol = n - 1;
        canReach = false;

        auto t = allgrid;
        dfs(t, n - 1, 0, mid, 0, canReach);
        if (canReach) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    cnt2 = left;

    cout << cnt1 + cnt2 << endl;
    return;
}