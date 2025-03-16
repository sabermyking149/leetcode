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
// 采用dijstra
int dij(vector<vector<char>>& grid, pair<int, int> start, pair<int, int> end, char sign)
{
    int i;
    int n = grid.size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int ans;
    // dist[i][j] - 从(0, 0)到(i, j)最少经过异色方块
    vector<vector<int>> dist(n, vector<int>(n, 0x3f3f3f3f));
    queue<pair<pair<int, int>, int>> q;

    if (grid[start.first][start.second] != sign) {
        q.push({start, 1});
    } else {
        q.push({start, 0});
    }
    ans = 0x3f3f3f3f;
    while (q.size()) {
        auto p = q.front();
        q.pop();
        if (dist[p.first.first][p.first.second] < p.second) {
            continue;
        }
        dist[p.first.first][p.first.second] = p.second;
        if (p.first == end) {
            ans = min(ans, p.second);
        }
        for (i = 0; i < 4; i++) {
            auto row = p.first.first + directions[i][0];
            auto col = p.first.second + directions[i][1];
            if (row < 0 || row >= n || col < 0 || col >= n) {
                continue;
            }
            if (p.second + (grid[row][col] == sign ? 0 : 1) < dist[row][col]) {
                dist[row][col] = p.second + (grid[row][col] == sign ? 0 : 1);
                q.push({{row, col}, dist[row][col]});
            }
        }
    }
    return ans;
}
void ARC_177_C_1()
{
    int i, j;
    int n, cnt1, cnt2;

    cin >> n;

    vector<vector<char>> grid(n, vector<char>(n, ' '));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            cin >> grid[i][j];
        }
    }

    cnt1 = dij(grid, {0, 0}, {n - 1, n - 1}, 'R');
    cnt2 = dij(grid, {n - 1, 0}, {0, n - 1}, 'B');

    cout << cnt1 + cnt2 << endl;
    return;
}


// 恶心人的输入格式
void ABC_356_C()
{
    int i, j, p;
    int N, M, K;
    int cnt, t;
    int c, a;
    char r;
    bool conflict = false;

    cin >> N >> M >> K;

    vector<vector<bool>> keys(1 << N, vector<bool>(N, false));

    cnt = 0;
    for (i = 0; i < (1 << N); i++) {
        j = 0;
        t = i;
        while (j < N) {
            keys[i][j] = t % 2;
            t /= 2;
            j++;
        }
    }
    vector<vector<int>> conditions(M);
    for (i = 0; i < M; i++) {
        cin >> c;
        for (j = 0; j < c; j++) {
            cin >> a;
            conditions[i].emplace_back(a);
        }
        cin >> r;
        if (r == 'o') {
            conditions[i].emplace_back(1);
        } else {
            conditions[i].emplace_back(0);
        }
    }

    for (i = 0; i < keys.size(); i++) {
        conflict = false;
        for (j = 0; j < M; j++) {
            t = 0;
            for (p = 0; p < conditions[j].size() - 1; p++) {
                if (keys[i][conditions[j][p] - 1]) {
                    t++;
                }
            }
            auto res = conditions[j][p];
            if ((t >= K && res == 0) || (t < K && res == 1)) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            cnt++;
        }
    }
    cout << cnt << endl;
}


void ABC_356_D()
{
    int i;
    int mod = 998244353;
    unsigned long long N, M;
    unsigned long long d, r, ans;

    cin >> N >> M;

    ans = 0;
    for (i = 0; i < 60; i++) {
        if ((M & (1ull << i)) == (1ull << i)) {
            d = N / (1ull << (i + 1));
            r = N % (1ull << (i + 1));
            ans = (ans + d * (1ull << i) + (r > (1ull << i) - 1 ? r - (1ull << i) + 1 : 0)) % mod;
        }
    }
    cout << ans << endl;
}


void ABC_357_C()
{
    int N;

    cin >> N;

    function<vector<string> (int, int)> CreateCarpet = [&CreateCarpet](int n, int color) {
        vector<string> ans;
        string t;
        if (n == 0) {
            if (color == 1) {
                t = "#";
                ans.emplace_back(t);
                return ans;
            }
            t = ".";
            ans.emplace_back(t);
            return ans;
        }
        vector<string> a = CreateCarpet(n - 1, color);
        // vector<string> b = CreateCarpet(n - 1, 1 - color);
        int size = a.size();

        ans.resize(size * 3);

        int i;
        for (i = 0; i < size; i++) {
            ans[i] = a[i] + a[i] + a[i];
        }
        int len = a[0].size();
        t.append(len, '.');
        for (i = size; i < size * 2; i++) {
            ans[i] = a[i - size] + t + a[i - size];
        }
        for (i = size * 2; i < size * 3; i++) {
            ans[i] = a[i - size * 2] + a[i - size * 2] + a[i - size * 2];
        }

        return ans;
    };

    int i;
    vector<string> ans = CreateCarpet(N, 1);
    int n = ans.size();

    for (i = 0; i < n; i++) {
        cout << ans[i] << endl;
    }
}


void ABC_364_E()
{
    int i, j, k;
    int n;
    int a, b;
    int x, y;
    int ans = 0;

    cin >> n >> x >> y;

    vector<vector<int>> dishes(n, vector<int>(2));
    for (i = 0; i < n; i++) {
        cin >> a >> b;
        dishes[i] = {a, b};
    }

    // dp[i][j][k] - 前i个物品选j个的x值对应的最小y值
    vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(n + 1, vector<int>(x + 1, 0x3f3f3f3f)));
    dp[0][0][0] = 0;
    for (i = 1; i <= n; i++) {
        for (j = 0; j <= i; j++) {
            if (j == 0) {
                dp[i][j][0] = 0;
                continue;
            }
            for (k = 1; k <= x; k++) {
                // 全选完
                if (j == i) {
                    if (k >= dishes[i - 1][0]) {
                        dp[i][j][k] = dp[i - 1][j - 1][k - dishes[i - 1][0]] + dishes[i - 1][1];
                    }
                    if (dp[i][j][k] <= y) {
                        ans = max(ans, j);
                    }
                    continue;
                }
                // 不全选完, 2种情况, 不选第i个; 选第i个
                dp[i][j][k] = min(dp[i][j][k], dp[i - 1][j][k]);
                if (k >= dishes[i - 1][0]) {
                    auto t = dp[i - 1][j - 1][k - dishes[i - 1][0]] + dishes[i - 1][1];
                    if (t <= y) {
                        dp[i][j][k] = min(dp[i][j][k], t);
                    }
                }
                if (dp[i][j][k] <= y) {
                    ans = max(ans, j);
                }
            }
        }
    }
    if (ans != n) {
        ans++;
    }
    cout << ans << endl;
}


void ABC_365_E()
{
    int i;
    int n;

    cin >> n;

    vector<int> a(n);
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }
}


// 三维前缀和
void ABC_366_D()
{
    int i, j, k;
    int n;

    cin >> n;

    int cube[101][101][101] = {0};
    int prefix[101][101][101] = {0};
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            for (k = 1; k <= n; k++) {
                cin >> cube[i][j][k];
                prefix[i][j][k] = cube[i][j][k] + 
                                  prefix[i - 1][j][k] +
                                  prefix[i][j - 1][k] -
                                  prefix[i - 1][j - 1][k] +
                                  prefix[i][j][k - 1] -
                                  prefix[i - 1][j][k - 1] -
                                  prefix[i][j - 1][k - 1] +
                                  prefix[i - 1][j - 1][k - 1];
            }
        }
    }

    int q;

    cin >> q;

    int Lx, Rx, Ly, Ry, Lz, Rz;
    int ret;
    vector<int> ans;
    for (i = 0; i < q; i++) {
        cin >> Lx >> Rx >> Ly >> Ry >> Lz >> Rz;
        ret = prefix[Rx][Ry][Rz] -
              prefix[Lx - 1][Ry][Rz] -
              prefix[Rx][Ly - 1][Rz] +
              prefix[Lx - 1][Ly - 1][Rz] -
              prefix[Rx][Ry][Lz - 1] +
              prefix[Lx - 1][Ry][Lz - 1] +
              prefix[Rx][Ly - 1][Lz - 1] -
              prefix[Lx - 1][Ly - 1][Lz - 1];
        ans.emplace_back(ret);
    }
    for (auto a : ans) {
        cout << a << endl;
    }
}


void ABC_370_D()
{
    int i, j, k;
    int h, w, q;

    cin >> h >> w >> q;

    int remain = h * w;
    //vector<int> walls(remain, 1);
    vector<int> walls;
    vector<set<int>> rows(h);
    vector<set<int, greater<int>>> rows_r(h);
    vector<set<int>> cols(w);
    vector<set<int, greater<int>>> cols_r(w);

    walls.assign(remain, 1);
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            rows[i].emplace(j);
            rows_r[i].emplace(j);
            cols[j].emplace(i);
            cols_r[j].emplace(i);
        }
    }

    int r, c;

    for (k = 0; k < q; k++) {
        cin >> r >> c;
        if (walls[(r - 1) * w + c - 1]) {
            remain--;
            walls[(r - 1) * w + c - 1] = 0;
            rows[r - 1].erase(c - 1);
            rows_r[r - 1].erase(c - 1);
            cols[c - 1].erase(r - 1);
            cols_r[c - 1].erase(r - 1);
        } else {
            auto it = rows[r - 1].upper_bound(c - 1);
            if (it != rows[r - 1].end()) {
                walls[(r - 1) * w + *it] = 0;
                rows[r - 1].erase(*it);
                rows_r[r - 1].erase(*it);
                cols[*it].erase(r - 1);
                cols_r[*it].erase(r - 1);
                remain--;
            }

            it = rows_r[r - 1].upper_bound(c - 1);
            if (it != rows_r[r - 1].end()) {
                walls[(r - 1) * w + *it] = 0;
                rows[r - 1].erase(*it);
                rows_r[r - 1].erase(*it);
                cols[*it].erase(r - 1);
                cols_r[*it].erase(r - 1);
                remain--;
            }

            it = cols[c - 1].upper_bound(r - 1);
            if (it != cols[c - 1].end()) {
                walls[*it * w + c - 1] = 0;
                cols[c - 1].erase(*it);
                cols_r[c - 1].erase(*it);
                rows[*it].erase(c - 1);
                rows_r[*it].erase(c - 1);
                remain--;
            }

            it = cols_r[c - 1].upper_bound(r - 1);
            if (it != cols_r[c - 1].end()) {
                walls[*it * w + c - 1] = 0;
                cols[c - 1].erase(*it);
                cols_r[c - 1].erase(*it);
                rows[*it].erase(c - 1);
                rows_r[*it].erase(c - 1);
                remain--;
            }
        }
        // cout << remain << endl;
    }
    cout << remain << endl;
}


// 从后往前单调递减栈
void ABC_372_D()
{
    int i;
    int n;
    stack<int> st;
    vector<int> ans;

    cin >> n;

    vector<int> h(n);
    for (i = 0; i < n; i++) {
        cin >> h[i];
    }
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            ans.emplace_back(0);
            st.push(i);
            continue;
        }
        auto idx = st.top();
        ans.emplace_back(st.size());
        if (h[idx] > h[i]) {
            st.push(i);
            continue;
        }
        while (h[idx] <= h[i]) {
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    reverse(ans.begin(), ans.end());
    for (i = 0; i < ans.size(); i++) {
        cout << ans[i] << (i == n - 1 ? "\n" : " ");
    }
}


void ABC_373_D()
{
    int i;
    int n, m;
    int from, to;
    long long w;

    cin >> n >> m;

    vector<vector<pair<int, long long>>> edgesWithWeight(n + 1);
    for (i = 0; i < m; i++) {
        cin >> from >> to >> w;
        edgesWithWeight[from].push_back({to, w});
        // 加上反向边和对应负权重方便dfs遍历
        edgesWithWeight[to].push_back({from, -w});
    }

    vector<long long> dist(n + 1, LLONG_MIN);
    function<void (int)> dfs = [&dfs, &edgesWithWeight, &dist](int cur) {
        for (auto it : edgesWithWeight[cur]) {
            if (dist[it.first] == LLONG_MIN) {
                dist[it.first] = dist[cur] + it.second;
                dfs(it.first);
            }
        }
    };
    for (i = 1; i <= n; i++) {
        if (dist[i] == LLONG_MIN) {
            dist[i] = 0;
            dfs(i);
        }
    }
    for (i = 1; i <= n; i++) {
        cout << dist[i] << (i == n ? "\n" : " ");
    }
}


void ABC_374_D()
{
    int i, j, k;
    int N;
    double S, T;

    cin >> N >> S >> T;
    vector<vector<double>> pos(N, vector<double>(4));

    for (i = 0; i < N; i++) {
        for (j = 0; j < 4; j++) {
            cin >> pos[i][j];
        }
    }
    vector<vector<int>> permutation;
    vector<int> record(N);
    vector<int> visited(N, false);
    function <void (int)> GeneratePermutation = [&GeneratePermutation, &record, &permutation, &visited](int idx) {
        if (idx == record.size()) {
            permutation.emplace_back(record);
            return;
        }
        int i;
        for (i = 0; i < record.size(); i++) {
            if (visited[i] == false) {
                visited[i] = true;
                record[idx] = i + 1;
                GeneratePermutation(idx + 1);
                visited[i] = false;
            }
        }
    };
    GeneratePermutation(0);
    /*for (auto per : permutation) {
        for (auto p : per) cout << p << " ";
        cout << endl;
    }*/

    auto getDist = [](double a, double b, double c , double d) {
        return sqrt((c - a) * (c - a) + (d - b) * (d - b));
    };

    double emittingTime = 0;
    for (auto p : pos) {
        emittingTime += getDist(p[0], p[1], p[2], p[3]) / T;
    }
    double ans = 0x3f3f3f3f * 1.0;
    double notEmittingTime;
    vector<int> bits(N);
    vector<double> curPos(2);
    for (i = 0; i < permutation.size(); i++) {
        for (k = 0; k < (1 << N); k++) {
            auto t = k;
            j = 0;
            bits.assign(N, 0);
            while (t) {
                bits[j] = t % 2;
                t /= 2;
                j++;
            }
            notEmittingTime = 0;
            curPos.assign(2, 0.0);
            for (j = 0; j < N; j++) {
                if (bits[j] == 0) {
                    notEmittingTime += getDist(curPos[0], curPos[1], pos[permutation[i][j] - 1][0], pos[permutation[i][j] - 1][1]) / S;
                    curPos[0] = pos[permutation[i][j] - 1][2];
                    curPos[1] = pos[permutation[i][j] - 1][3];
                } else {
                    notEmittingTime += getDist(curPos[0], curPos[1], pos[permutation[i][j] - 1][2], pos[permutation[i][j] - 1][3]) / S;
                    curPos[0] = pos[permutation[i][j] - 1][0];
                    curPos[1] = pos[permutation[i][j] - 1][1];
                }
            }
            ans = min(ans, emittingTime + notEmittingTime);
        }
    }
    cout << setprecision(16) << ans << endl;
}


void ABC_375_C()
{
    int i;
    int n;

    cin >> n;
    vector<string> grid(n);

    for (i = 0; i < n; i++) {
        cin >> grid[i];
    }

    // 坐标变换 [x, y] -> [y, N - 1 - x] -> [N - 1 - x, N - 1 - y] -> [N - 1 - y, x]
    auto rotateEdge = [&grid](int r, int cnt) {
        cnt %= 4;
        if (cnt == 0) {
            return;
        }

        int j;
        int n = grid.size();

        while (cnt) {
            for (j = r; j < n - 1 - r; j++) {
                auto top = grid[r][j];
                grid[r][j] = grid[n - 1 - j][r];
                grid[n - 1 - j][r] = grid[n - 1 - r][n - 1 - j];
                grid[n - 1 - r][n - 1 - j] = grid[j][n - 1 - r];
                grid[j][n - 1 - r] = top;
            }
            cnt--;
        }
    };

    for (i = 0; i < n / 2; i++) {
        rotateEdge(i, i + 1);
    }
    for (auto g : grid) {
        cout << g << endl;
    }
}


void ABC_375_D()
{
    int i;
    int n;
    string s;

    cin >> s;

    n = s.size();
    vector<int> chIdx(26, -1);
    vector<int> cnt(26, 0);
    vector<long long> sum(26, 0);
    long long ans = 0;
    for (i = 0; i < n; i++) {
        if (chIdx[s[i] - 'A'] == -1) {
            chIdx[s[i] - 'A'] = i;
            cnt[s[i] - 'A']++;
        } else {
            auto idx = chIdx[s[i] - 'A'];
            sum[s[i] - 'A'] += (i - idx) * static_cast<long long>(cnt[s[i] - 'A']);  
            ans += sum[s[i] - 'A'] - cnt[s[i] - 'A'];
            cnt[s[i] - 'A']++;
            chIdx[s[i] - 'A'] = i;
        }
    }
    cout << ans << endl;
}


// 有向图包含某节点的最小环
void ABC_376_D()
{
    int i;
    int n, m;
    int from, to;
    int INF = 0x3f3f3f3f;

    cin >> n >> m;

    vector<vector<pair<int, int>>> edgeWithWeight(n + 1);
    // 权重记为1
    for (i = 0; i < m; i++) {
        cin >> from >> to;
        edgeWithWeight[from].push_back({to, 1});
    }

    vector<int> dist(n + 1, INF);
    queue<pair<int, int>> q;

    // 用dijkstra间接求最小环
    q.push({1, 0});
    while (q.size()) {
        auto t = q.front();
        q.pop();

        if (dist[t.first] < t.second) {
            continue;
        }
        dist[t.first] = t.second;
        for (auto it : edgeWithWeight[t.first]) {
            // 发现过节点1的环
            if (it.first == 1) {
                // cout << t.first << " " << dist[t.first] << endl;
                if (dist[it.first] == 0) {
                    dist[it.first] = t.second + 1;
                } else {
                    dist[it.first] = min(dist[it.first], t.second + 1);
                }
                // cout << dist[it.first] << endl;
                continue;
            }
            if (dist[t.first] + it.second < dist[it.first]) {
                dist[it.first] = dist[t.first] + it.second;
                q.push({it.first, dist[it.first]});
            }
        }
    }
    cout << (dist[1] == 0 ? -1 : dist[1]) << endl;
}


void ABC_377_D()
{
    int i;
    int n, m;
    int l, r;
    long long ans;

    cin >> n >> m;
    // left[i] - [left[i], i]最大不重叠区间
    vector<int> left(m + 1, 1);
    for (i = 0; i < n; i++) {
        cin >> l >> r;
        left[r] = max(l + 1, left[r]);
    }
    ans = 0;
    for (i = 1; i <= m; i++) {
        left[i] = max(left[i], left[i - 1]);
        ans += i - left[i] + 1;
    }

    cout << ans << endl;
}


void ABC_378_D()
{
    int i, j;
    int ans;
    int H, W, K;

    cin >> H >> W >> K;

    vector<vector<char>> grid(H, vector<char>(W));
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            cin >> grid[i][j];
        }
    }

    // dp[i][j][k] - 移动到(i, j)且刚好走了k步的方法数
    /*vector<vector<vector<long long>>> dp(H, vector<vector<long long>>(W, vector<long long>(K + 1, 0)));
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            dp[i][j][0] = 1;
        }
    }*/

    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    vector<vector<int>> visited(H, vector<int>(W, 0));
    ans = 0;
    function<void (int, int, int)> dfs = [&dfs, &directions, &grid, &visited, &K, &ans](int row, int col, int cnt) {
        visited[row][col] = 1;
        if (cnt == K) {
            ans++;
            visited[row][col] = 0;
            return;
        }
        for (int i = 0; i < 4; i++) {
            auto nr = row + directions[i][0];
            auto nc = col + directions[i][1];
            if (nr < 0 || nr >= grid.size() || nc < 0 || 
                nc >= grid[0].size() || visited[nr][nc] || grid[nr][nc] == '#') {
                continue;
            }
            dfs(nr, nc, cnt + 1);
        }
        visited[row][col] = 0;
    };
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            if (grid[i][j] == '#') {
                continue;
            }
            dfs(i, j, 0);
        }
    }
    cout << ans << endl;
}


void ABC_382_D()
{
    int i;
    int n, m;

    cin >> n >> m;

    int a = m - (n - 1) * 10;
    vector<vector<int>> res;
    vector<int> record;
    function<void (int, int, int)> Generate = [&Generate, &res, &record, &n, &m](int start, int cur, int cnt) {
        if (cnt == n) {
            res.emplace_back(record);
            return;
        }
        int i;
        int diff = (m - start) / (n - 1) + 10;
        for (i = 10; i <= diff; i++) {
            if (cur + i <= m && cur + i + (n - 1 - cnt) * 10 <= m) {
                record.emplace_back(cur + i);
                Generate(start, cur + i, cnt + 1);
                record.pop_back();
            } else {
                break;
            }
        }
    };
    for (i = 1; i <= a; i++) {
        record.emplace_back(i);
        Generate(i, i, 1);
        record.pop_back();
    }
    cout << res.size() << endl;
    for (auto v : res) {
        for (i = 0; i < v.size(); i++) {
            cout << v[i] << (i != v.size() - 1 ? " " : "\n");
        }
    }
}


void ABC_386_D()
{
    int i;
    int n, m;

    cin >> n >> m;

    int x, y;
    char color;
    vector<vector<int>> pos;
    map<int, int> edge;
    for (i = 0; i < m; i++) {
        cin >> x >> y >> color;
        if (color == 'W') {
            if (edge.count(x)) {
                edge[x] = min(edge[x], y);
            } else {
                edge[x] = y;
            }
        } else {
            pos.push_back({x, y});
        }
    }
    if (edge.empty() || pos.empty()) {
        cout << "Yes\n";
        return;
    }

    int cur = edge.begin()->second;
    for (auto it : edge) {
        edge[it.first] = min(edge[it.first], cur);
        cur = edge[it.first];
    }
    for (auto p : pos) {
        auto it = edge.lower_bound(p[0]);
        if (it == edge.end()) {
            it--;
            if (p[1] >= it->second) {
                cout << "No\n";
                return;
            }
        } else {
            if (it == edge.begin()) {
                if (p[0] == it->first && p[1] >= it->second) {
                    cout << "No\n";
                    return;
                }
            } else {
                auto prev = it;
                prev--;
                if ((p[0] == it->first && p[1] >= it->second) || 
                    (p[0] < it->first && p[1] >= prev->second)) {
                    cout << "No\n";
                    return;
                }
            }
        }
    }
    cout << "Yes\n";
}


void ABC_388_D()
{
    int i;
    int n;

    cin >> n;

    vector<int> v(n);
    for (i = 0; i < n; i++) {
        cin >> v[i];
    }

    vector<int> diff(n + 1); // diff[i] = v[i] - v[i - 1]
    for (i = 0; i < n; i++) {
        if (i != 0) {
            diff[i] += diff[i - 1];
            v[i] += diff[i];
        }
        auto t = min(n - 1 - i, v[i]);
        diff[i + 1]++;
        diff[i + 1 + t]--;
        v[i] -= t;
    }

    for (i = 0; i < n; i++) {
        cout << v[i] << (i == n - 1 ? "\n" : " ");
    }
}


void ABC_388_E()
{
    int i, j;
    int n;

    cin >> n;
    vector<int> v(n);
    for (i = 0; i < n; i++) {
        cin >> v[i];
    }

    int left, right, mid;

    left = 0;
    right = n - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        i = 0;
        j = mid + 1;
        while (i <= mid && j < n) {
            if (v[i] * 2 <= v[j]) {
                i++;
                j++;
            } else {
                j++;
            }
        }
        if (i == mid + 1) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    cout << right + 1 << endl;
}


// 正方形四个顶点是否在圆(0.5, 0.5, r)内
bool IsInside(double x, double y, double r)
{
    return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - r * r < 1e-6;
}
void ABC_389_D()
{
    int i;
    int r;
    int left, right, mid;
    long long a, b;
    long long ans = 0;

    cin >> r;
    if (r == 1) {
        cout << "1" << endl;
        return;
    }
    for (i = -r + 1; i <= r; i++) {
        left = -r;
        right = r;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (IsInside(mid, i, r) && IsInside(mid - 1, i, r) &&
                IsInside(mid, i + 1, r) && IsInside(mid - 1, i + 1, r)) {
                right = mid - 1;        
            } else {
                left = mid + 1;
            }
        }
        if (left > r) {
            continue;
        }
        a = left;
        
        left = 0;
        right = r;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (IsInside(mid, i, r) && IsInside(mid - 1, i, r) &&
                IsInside(mid, i + 1, r) && IsInside(mid - 1, i + 1, r)) {
                left = mid + 1;
                // printf ("1: %d %d %d\n", i, a, left);
            } else {
                right = mid - 1;
            }
        }
        // printf ("%d %d %d\n", i, a, right);
        b = right;
        ans += b - a + 1;
    }
    cout << ans << endl;
}


void ABC_391_D()
{
    int i;
    int x, y;
    int n, w, q, t, a;

    cin >> n >> w;

    vector<vector<pair<int, int>>> pos(w + 1);
    for (i = 1; i <= n; i++) {
        cin >> x >> y;
        pos[x].push_back({y, i});
    }
    for (i = 1; i <= w; i++) {
        sort(pos[i].begin(), pos[i].end());
    }
    vector<int> finalTime(n + 1, 0x3f3f3f3f); // 每个block最后落下的时间
    int idx = 0;
    int maxTime;
    bool stable = false;
    while (1) {
        stable = false;
        maxTime = 0;
        for (i = 1; i <= w; i++) {
            if (pos[i].size() < idx + 1) { // 保持稳定, 以后都不会消掉
                // printf ("%d\n", i);
                stable = true;
                break;
            }
            maxTime = max(maxTime, pos[i][idx].first);
        }
        if (stable) {
            break;
        }
        for (i = 1; i <= w; i++) {
            finalTime[pos[i][idx].second] = maxTime;
        }
        idx++;
    }
    cin >> q;
    for (i = 0; i < q; i++) {
        cin >> t >> a;
        if (finalTime[a] > t) {
            cout << "Yes\n";
        } else {
            cout << "No\n";
        }
    }
}


struct TriTreeNode {
    int val;
    int no; // 编号
    TriTreeNode *left;
    TriTreeNode *mid;
    TriTreeNode *right;
    TriTreeNode(int val, int no) : val(val), no(no), left(nullptr), mid(nullptr), right(nullptr) {};
    ~TriTreeNode() {
        delete left;
        delete mid;
        delete right;
    }
};
vector<string> trans(string& s)
{
    int i;
    int n;
    int cnt0, cnt1;
    string t;
    vector<string> res;

    res.emplace_back(s);
    while (1) {
        if (s.size() < 3) {
            break;
        }
        t.clear();
        n = s.size();
        for (i = 0; i < n; i += 3) {
            cnt0 = cnt1 = 0;
            if (s[i] == '0') {
                cnt0++;
            } else {
                cnt1++;
            }
            if (s[i + 1] == '0') {
                cnt0++;
            } else {
                cnt1++;
            }
            if (s[i + 2] == '0') {
                cnt0++;
            } else {
                cnt1++;
            }
            if (cnt0 > cnt1) {
                t.append(1, '0');
            } else {
                t.append(1, '1');
            }
        }
        res.emplace_back(t);
        s = t;
    }
    return res;
}
void ABC_391_E()
{
    int j;
    int n, m, layer, no;
    string s;

    cin >> n >> s;

    vector<string> res = trans(s);
    unordered_map<TriTreeNode *, int> nodesNo;
    reverse(res.begin(), res.end());
    n = res.size();
    // 构建3叉树
    TriTreeNode *root = new TriTreeNode(res[0][0] - '0', 0);
    nodesNo[root] = 0;
    queue<TriTreeNode *> q;
    q.push(root);
    layer = 1;
    no = 1;
    while (!q.empty()) {
        if (layer == n) {
            break;
        }
        m = q.size();
        for (j = 0; j < m; j++) {
            auto node = q.front();
            q.pop();
            node->left = new TriTreeNode(res[layer][j * 3] - '0', no);
            q.push(node->left);
            nodesNo[node->left] = no;
            no++;

            node->mid = new TriTreeNode(res[layer][j * 3 + 1] - '0', no);
            q.push(node->mid);
            nodesNo[node->mid] = no;
            no++;

            node->right = new TriTreeNode(res[layer][j * 3 + 2] - '0', no);
            q.push(node->right);
            nodesNo[node->right] = no;
            no++;
        }
        layer++;
    }
    // dp(node, i) - 将node值改变成i的最小操作次数
    vector<vector<int>> dp(no, vector<int>(2, -1));
    function<int (TriTreeNode *, int)> dfs = [&dfs, &dp, &nodesNo](TriTreeNode *node, int val) {
        if (dp[nodesNo[node]][val] != -1) {
            return dp[nodesNo[node]][val];
        }
        if (node->left == nullptr && node->mid == nullptr && node->right == nullptr) {
            if (node->val == val) {
                dp[nodesNo[node]][val] = 0;
            } else {
                dp[nodesNo[node]][val] = 1;
            }
            return dp[nodesNo[node]][val];
        }
        int ans = 0x3f3f3f3f;
        if (val == 0) {
            ans = min({ans, dfs(node->left, 0) + dfs(node->mid, 0) + dfs(node->right, 0),
                            dfs(node->left, 1) + dfs(node->mid, 0) + dfs(node->right, 0),
                            dfs(node->left, 0) + dfs(node->mid, 1) + dfs(node->right, 0),
                            dfs(node->left, 0) + dfs(node->mid, 0) + dfs(node->right, 1)});
        } else {
            ans = min({ans, dfs(node->left, 1) + dfs(node->mid, 1) + dfs(node->right, 1),
                            dfs(node->left, 1) + dfs(node->mid, 1) + dfs(node->right, 0),
                            dfs(node->left, 0) + dfs(node->mid, 1) + dfs(node->right, 1),
                            dfs(node->left, 1) + dfs(node->mid, 0) + dfs(node->right, 1)});
        }
        dp[nodesNo[node]][val] = ans;
        return ans;
    };
    int ans = dfs(root, 1 - root->val);
    cout << ans << endl;
    delete root;
}


void ABC_393_D()
{
    int i;
    int n, idx;
    string s;
    int cnt1, cnt0, mid;

    cin >> n >> s;
    cnt1 = 0;
    for (i = 0; i < n; i++) {
        if (s[i] == '1') {
            cnt1++;
        }
    }
    mid = (cnt1  + 1) / 2;
    cnt1 = 0;
    for (i = 0; i < n; i++) {
        if (s[i] == '1') {
            cnt1++;
            if (cnt1 == mid) {
                idx = i;
                break;
            }
        }
    }
    long long ans = 0;
    cnt0 = 0;
    for (i = idx - 1; i >= 0; i--) {
        if (s[i] == '0') {
            cnt0++;
        } else {
            ans += cnt0;
        }
    }
    cnt0 = 0;
    for (i = idx + 1; i < n; i++) {
        if (s[i] == '0') {
            cnt0++;
        } else {
            ans += cnt0;
        }
    }
    cout << ans << endl;
}


void ABC_393_E()
{
    int i, j;
    int n, k, t;
    int cur;
    unordered_map<int, int> cnt;

    cin >> n >> k;

    vector<int> nums(n);
    for (i = 0; i < n; i++) {
        cin >> nums[i];
        cnt[nums[i]]++;
    }

    int d = *max_element(nums.begin(), nums.end());
    // ans[i] - 包含i的k个最大gcd
    vector<int> ans(d + 1, 1);

    for (i = 1; i <= d; i++) {
        j = 1;
        t = i;
        cur = 0;
        while (1) {
            t = i * j;
            if  (t > d) {
                break;
            }
            if (cnt.count(t)) {
                cur += cnt[t];
            }
            j++;
        }
        // i的倍数一定都有公约数i
        if (cur >= k) {
            t = i;
            j = 1;
            while (1) {
                t = i * j;
                if (t > d) {
                    break;
                }
                ans[t] = i;
                j++;
            }
        }
    }
    for (auto n : nums) {
        cout << ans[n] << endl;
    }
}


void ABC_397_D()
{
    long long i;
    long long n;
    cin >> n;

    // x^3 - y^3 = (x - y)[(x - y)^2 + 3xy] = n; 如果n能分解成 a * b 那么必然 x - y 对应小的那个因子
    // 且 x - y 的范围在[1, n^(1/3)]
    long long diff, multiply;
    for (i = 1; i <= 1e6; i++) {
        if (n % i != 0) {
            continue;
        }
        if (n / i <= i * i || (n / i - i * i) % 3 != 0) {
            continue;
        }
        diff = i;
        multiply = (n / i - i * i) / 3;
        // 由 x - y 和 xy 凑 x + y
        long long sum = sqrt(diff * diff + multiply * 4);
        if (sum * sum == diff * diff + multiply * 4) {
            cout << (sum + diff) / 2 << " " << (sum - diff) / 2 << endl;
            return;
        }
    }
    cout << -1 << endl;
}