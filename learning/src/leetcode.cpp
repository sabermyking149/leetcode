#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>
#include <iterator>
#include <algorithm>
#include <queue>
#include <cmath>

#include "pub.h"

using namespace std;

int gcd(int a, int b)
{
    if(b == 0) {
        return a;
    }
    return gcd(b, a % b);
}
void initVisited(vector<int>& visited, int val)
{
    for (int i = 0; i < visited.size(); i++) {
        visited[i] = val;
    }
}
void CreateLoop(int cur, vector<int>& visited, unordered_map<int, int> relative, bool &findEnd, unordered_set<int>& loopNode)
{
    if (findEnd) {
        return;
    }
    if (loopNode.find(cur) != loopNode.end()) {
        findEnd = true;
        return;
    }
    if (visited[cur] == 1) {
        loopNode.emplace(cur);
    }
    CreateLoop(relative[cur], visited, relative, findEnd, loopNode);
}
void DFSFindLoop(unordered_map<int, unordered_set<int>>& e, int cur, int time, unordered_map<int, int>  &visitedTime,
    vector<int>& visited, bool &findLoop, int &loopSize)
{
    if (findLoop) {
        return;
    }
    if (visited[cur] == 1) {
        findLoop = true;
        loopSize = time - visitedTime[cur];
        return;
    }
    visited[cur] = 1;
    visitedTime[cur] = time;
    if (e.find(cur) != e.end()) {
        for (auto it : e[cur]) {
            if (visited[it] != 2) {
                DFSFindLoop(e, it, time + 1, visitedTime, visited, findLoop, loopSize);
            }
        }
    }
    visited[cur] = 2;
}
int longestCycle(vector<int>& edges)
{
    int i;
    unordered_map<int, unordered_set<int>> e;

    for (i = 0; i < edges.size(); i++) {
        if (edges[i] != -1) {
            e[i].emplace(edges[i]);
        }
    }
    int n = edges.size();
    vector<int> visited(n, 0);
    unordered_map<int, int> visitedTime;
    int loopSize = 0;
    bool findLoop = false;

    int ans = -1;
    for (i = 0; i < n; i++) {
        findLoop = false;
        loopSize = -1;
        DFSFindLoop(e, i, 0, visitedTime, visited, findLoop, loopSize);
        ans = max(ans, loopSize);
    }
    return ans;
}

vector<TreeNode*> BSTs;
vector<TreeNode*> CreateBSTs(int start, int end)
{
    int i, j, k;
    vector<TreeNode*> trees;
    if (start > end) {
        return {nullptr};
    }
    for (i = start; i <= end; i++) {
        vector<TreeNode *> leftTree = CreateBSTs(start, i - 1);
        vector<TreeNode *> rightTree = CreateBSTs(i + 1, end);
        // cout << leftTree.size() << " " << rightTree.size() << endl;
        for (j = 0; j < leftTree.size(); j++) {
            for (k = 0; k < rightTree.size(); k++) {
                TreeNode *node = new TreeNode(i);
                node->left = leftTree[j];
                node->right = rightTree[k];
                trees.emplace_back(node);
            }
        }
    }
    return trees;
}
vector<TreeNode*> generateTrees(int n)
{
    BSTs = CreateBSTs(1, n);
    return BSTs;
}


vector<string> words;
unordered_map<string, int> um;
vector<string> ans;
int Find_s(string& a, int idx)
{
    if (words[idx] == a) {
        return idx;
    }
    int id = Find_s(words[idx], um[words[idx]]);
    words[idx] = words[id];
    return id;
}
void Union_s(string& a, string& b)
{
    int x, y;

    x = Find_s(a, um[a]);
    y = Find_s(b, um[b]);
    if (x != y) {
        words[y] = words[x];
    }
}
vector<string> MySplit(string& s, char separate)
{
    int i;
    int n = s.size();
    vector<string> ans;
    int cur = 0;

    for (i = 0; i < n; i++) {
        if (s[i] == separate) {
            ans.emplace_back(s.substr(cur, i - cur));
            cur = i + 1;
        }
    }
    ans.emplace_back(s.substr(cur, n - cur));
    return ans;
}
void DFSCreateSentences(vector<string>& texts, int curIdx, unordered_map<string, unordered_set<string>>& synonymsWords,
    vector<string>& words)
{
    if (curIdx == texts.size()) {
        string str;
        for (auto w : words) {
            str += w + ' ';
        }
        str[str.size() - 1] = '\0';
        ans.emplace_back(str);
        return;
    }
    if (synonymsWords.find(texts[curIdx]) == synonymsWords.end()) {
        words.push_back(texts[curIdx]);
        DFSCreateSentences(texts, curIdx + 1, synonymsWords, words);
        words.pop_back();
    } else {
        for (auto it : synonymsWords[texts[curIdx]]) {
            words.push_back(it);
            DFSCreateSentences(texts, curIdx + 1, synonymsWords, words);
            words.pop_back();
        }
    }
}
vector<string> generateSentences(vector<vector<string>>& synonyms, string text)
{
    int i;
    unordered_set<string> dict;

    for (auto s : synonyms) {
        dict.insert(s[0]);
        dict.insert(s[1]);
    }

    int n = dict.size();
    for (auto it : dict) {
        words.emplace_back(it);
        um[it] = words.size() - 1;
    }
    vector<string> t = words;

    for (auto s : synonyms) {
        Union_s(s[0], s[1]);
    }
    for (i = 0; i < words.size(); i++) {
        words[i] = words[Find_s(words[i], um[words[i]])];
    }
    unordered_map<string, unordered_set<string>> synonymsWords;
    for (i = 0; i < words.size(); i++) {
        synonymsWords[words[i]].insert(t[i]);
    }
    for (auto it : synonymsWords) {
        for (auto it1 : it.second) {
            if (synonymsWords.find(it1) == synonymsWords.end()) {
                synonymsWords[it1] = it.second;
            }
        }
    }
    vector<string> texts = MySplit(text, ' ');
    vector<string> strs;
    DFSCreateSentences(texts, 0, synonymsWords, strs);
    sort(ans.begin(), ans.end());
    return ans;
}

vector<string> trulyMostPopular(vector<string>& names, vector<string>& synonyms)
{
    unordered_set<string> dict;
    string s1, s2;
    int i;
    int idx;
    int num;
    vector<pair<string, string>> synonymsPair;
    for (auto s : synonyms) {
        idx = s.find(",");
        s1 = s.substr(1, idx - 1);
        s2 = s.substr(idx + 1, s.size() - 1 - idx - 1);
        synonymsPair.emplace_back(make_pair(s1, s2));
        dict.insert(s1);
        dict.insert(s2);
    }
    // vector<string> words;
    // 对synonyms的处理
    for (auto it : dict) {
        words.emplace_back(it);
        um[it] = words.size() - 1; // 单词坐标
    }
    // 对names的处理
    vector<pair<string, int>> namesData;
    for (auto name: names) {
        idx = name.find("(");
        s1 = name.substr(0, idx);
        num = atoi(name.substr(idx + 1).c_str()); // 多截取了一个')',atoi会忽略
        namesData.emplace_back(make_pair(s1, num));
        if (um.find(s1) == um.end()) {
            words.emplace_back(s1);
            um[s1] = words.size() - 1;
        }
    }
    vector<string> t = words;
    for (auto sp : synonymsPair) {
        Union_s(sp.first, sp.second);
    }
    for (i = 0; i < words.size(); i++) {
        words[i] = words[Find_s(words[i], um[words[i]])];
    }

    unordered_map<string, set<string>> synonymsWordsDict;
    unordered_map<string, string> root; // 查并集的根
    for (i = 0; i < words.size(); i++) {
        synonymsWordsDict[words[i]].insert(t[i]);
        root[t[i]] = words[i];
    }
    unordered_map<string, int> collvalue;
    for (auto data : namesData) {
        collvalue[*synonymsWordsDict[root[data.first]].begin()] += data.second;
    }
    vector<string> ans;
    for (auto it : collvalue) {
        ans.emplace_back(it.first + "(" + to_string(it.second) + ")");
    }
    return ans;
}

int minTotalDistance(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    vector<int> row, col;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                row.emplace_back(i);
                col.emplace_back(j);
            }
        }
    }
    sort(row.begin(), row.end());
    sort(col.begin(), col.end());
    int x, y;

    x = row[row.size() / 2];
    y = row[col.size() / 2];
    int ans = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                ans += abs(i - x) + abs(j - y);
            }
        }
    }
    return ans;
}

bool IsPalindrome(string &s)
{
    int i, j;
    int n = s.size();

    i = 0;
    j = n - 1;
    while (i < j) {
        if (s[i] != s[j]) {
            return false;
        }
        i++;
        j--;
    }
    return true;
}

void PartPalindrome(string s, vector<string>& record, vector<vector<string>>& ans)
{
    int i;
    int n = s.size();

    if (s.size() == 0) {
        ans.emplace_back(record);
        return;
    }
    for (i = 0; i < n; i++) {
        string t = s.substr(0, i + 1);
        if (IsPalindrome(t)) {
            record.emplace_back(t);
            PartPalindrome(s.substr(i + 1), record, ans);
            record.pop_back();
        }
    }
}
vector<vector<string>> partition(string s)
{
    vector<vector<string>> ans;
    vector<string> record;
    PartPalindrome(s, record, ans);
    return ans;
}



class CMPMostBooked {
public:
    bool operator() (const vector<long long> &a, const vector<long long>& b)
    {
        if (a[1] != b[1]) {
            return a[1] > b[1];
        }
        return a[2] > b[2]; // 结束时间相同,返回编号小的房间
    }
};
int mostBooked(int n, vector<vector<int>>& meetings)
{
    int i;
    priority_queue<int, vector<int>, greater<int>> emptyRooms;
    priority_queue<vector<long long>, vector<vector<long long>>, CMPMostBooked> meetingTimes;
    vector<int> roomUse(n, 0);

    for (i = 0; i < n; i++) {
        emptyRooms.push(i);
    }
    sort(meetings.begin(), meetings.end());
    int emptyRoom = emptyRooms.top();
    emptyRooms.pop();
    roomUse[emptyRoom]++;

    meetingTimes.push({meetings[0][0], meetings[0][1], emptyRoom});
    for (i = 1; i < meetings.size(); i++) {
        vector<long long> t;
        while (meetingTimes.size()) {
            t = meetingTimes.top();
            if (t[1] <= meetings[i][0]) {
                emptyRoom = t[2];
                emptyRooms.push(emptyRoom);
                meetingTimes.pop();
            } else {
                break;
            }
        }
        if (meetingTimes.empty()) {
            emptyRoom = emptyRooms.top();
            emptyRooms.pop();

            roomUse[emptyRoom]++;
            meetingTimes.push({meetings[i][0], meetings[i][1], emptyRoom});
            continue;
        }
        if (emptyRooms.size() > 0) {
            emptyRoom = emptyRooms.top();
            emptyRooms.pop();

            roomUse[emptyRoom]++;
            meetingTimes.push({meetings[i][0], meetings[i][1], emptyRoom});
        } else {
            emptyRoom = t[2];
            roomUse[emptyRoom]++;
            meetingTimes.pop();
            vector<long long> newMeetings(3);
            newMeetings[0] = t[1];
            newMeetings[1] = t[1] + meetings[i][1] - meetings[i][0];
            newMeetings[2] = emptyRoom;
            meetingTimes.push(newMeetings);
        }
    }
    int maxUse = 0;
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (roomUse[i] > maxUse) {
            maxUse = roomUse[i];
            ans = i;
        }
    }
    return ans;
}


int kSimilarity(string s1, string s2)
{
    int i, j, k;
    int n, m;
    int step;
    unordered_set<string> visited;

    if (s1 == s2) {
        return 0;
    }
    m = s1.size();
    string ss1, ss2;
    for (i = 0; i < m; i++) {
        if (s1[i] != s2[i]) {
            ss1 += s1[i];
            ss2 += s2[i];
        }
    }

    queue<pair<string, int>> q;
    step = 0;
    m = ss1.size();
    q.push({ss1, 0});
    visited.emplace(ss1);
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            pair<string, int> p = q.front();
            string t = p.first;
            int idx = p.second;
            q.pop();
            if (t == ss2) {
                return step;
            }
            while (idx < m && t[idx] == ss2[idx]) {
                idx++;
            }
            for (j = idx; j < m; j++) {
                if (t[j] == ss2[idx] && t[j] != t[idx]) {
                    string tmp = t;
                    swap(tmp[j], tmp[idx]);
                    if (visited.find(tmp) == visited.end()) {
                        q.push({tmp, idx + 1});
                        visited.emplace(tmp);
                    }
                }
            }
        }
        step++;
    }
    return -1;
}


vector<int> GoThrough(vector<vector<int>>& maze, vector<int>& curPos, int len, int direction, vector<vector<int>>& pathLen)
{
    int row = maze.size();
    int col = maze[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    auto t = curPos;
    int dist = 0;
    while (1) {
        t[0] += directions[direction][0];
        t[1] += directions[direction][1];
        dist++;
        if (t[0] < 0 || t[0] >= row || t[1] < 0 || t[1] >= col || maze[t[0]][t[1]] == 1) {
            t[0] -= directions[direction][0];
            t[1] -= directions[direction][1];
            dist--;
            break;
        }
    }
    if (pathLen[t[0]][t[1]] > len + dist) {
        pathLen[t[0]][t[1]] = len + dist;
        return t;
    }
    return {-1, -1};
}
int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination)
{
    int i, j, k, d;
    int n;
    int row = maze.size();
    int col = maze[0].size();
    vector<vector<int>> pathLen(row, vector<int>(col, INT_MAX));
    queue<vector<int>> q;

    q.push(start);
    pathLen[start[0]][start[1]] = 0;
    while (q.size()) {
        n = q.size();
        for (k = 0; k < n; k++) {
            auto v = q.front();
            q.pop();

            /* if (v == destination) {
                return pathLen[v[0]][v[1]];
            } */
            for (i = 0; i < 4; i++) {
                auto t = GoThrough(maze, v, pathLen[v[0]][v[1]], i, pathLen);
                if (t[0] != -1) {
                    q.push(t);
                }
            }
        }
    }
    return pathLen[destination[0]][destination[1]] == INT_MAX ? -1 : pathLen[destination[0]][destination[1]];
}


class Solution {
public:
    vector<vector<int>> ans;
    void BallsRunning(vector<string>& plate, int row, int col, int direction, int step)
    {
        int a, b;
        int r = plate.size();
        int c = plate[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        a = row;
        b = col;
        while (step) {
            row += directions[direction][0];
            col += directions[direction][1];
            if (row >= r || row < 0 || col >= c || col < 0) {
                break;
            }
            if (plate[row][col] == 'O') {
                ans.push_back({a, b});
                break;
            }
            if (plate[row][col] == 'E') {
                direction = (direction + 1) % 4;
            } else if (plate[row][col] == 'W') {
                direction = (direction + 3) % 4;
            }
            step--;
        }
    }
    vector<vector<int>> ballGame(int num, vector<string>& plate)
    {
        int i, j;
        int row = plate.size();
        int col = plate[0].size();

        for (i = 1; i < row - 1; i++) {
            if (plate[i][0] == '.') {
                BallsRunning(plate, i, 0, 0, num);
            }
            if (plate[i][col - 1] == '.') {
                BallsRunning(plate, i, col - 1, 2, num);
            }
        }
        for (j = 1; j < col - 1; j++) {
            if (plate[0][j] == '.') {
                BallsRunning(plate, 0, j, 1, num);
            }
            if (plate[row - 1][j] == '.') {
                BallsRunning(plate, row - 1, j, 3, num);
            }
        }
        return ans;
    }
};
#if 0
static bool Cmp(pair<string, int>& a, pair<string, int>& b)
{
    /*if (a.second != b.second) {
        return a.second > b.second;
    }*/
    return a.second > b.second;

}
#endif
/*
vector<string> sortPeople(vector<string>& names, vector<int>& heights)
{
    vector<pair<string, int>> vp;
    int i;
    int n = names.size();
    for (i = 0; i < n; i++) {
        vp.push_back(make_pair(names[i], heights[i]));
    }

    //sort(vp.begin(), vp.end(), Cmp);
    vector<string> ans;
    for (auto p : vp) {
        ans.emplace_back(p.second);
    }
    return ans;
}
*/

unordered_set<string> trees;
unordered_set<string> t;
vector<TreeNode *> res;

string PreorderTree(TreeNode *node)
{
    string tree;
    if (node == nullptr) {
        return "";
    }
    tree = to_string(node->val) + "(" + PreorderTree(node->left) + ")" + "(" + PreorderTree(node->right) + ")";
    if (trees.find(tree) != trees.end() && t.find(tree) == t.end()) {
        t.emplace(tree);
        res.emplace_back(node);
    }
    trees.emplace(tree);
    return tree;
}
vector<TreeNode*> lightDistribution(TreeNode* root)
{
    PreorderTree(root);
    return res;
}


vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements)
{
    int i;
    int n = increase.size();
    int size;

    vector<int> v(3, 0);
    vector<vector<int>> prefixSum;

    prefixSum.emplace_back(v);
    for (i = 0; i < n; i++) {
        size = prefixSum.size();
        prefixSum.push_back({prefixSum[size - 1][0] + increase[i][0],
            prefixSum[size - 1][1] + increase[i][1],
            prefixSum[size - 1][2] + increase[i][2]});
    }
    int left, right, mid;
    int j;
    n = requirements.size();
    vector<int> ans(n, 0);
    for (i = 0; i < n; i++) {
        for (j = 0; j < 3; j++) {
            left = 0;
            right = prefixSum.size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (prefixSum[mid][j] >= requirements[i][j]) { // right + 1;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // printf ("left = %d, right = %d\n", left, right);
            if (left >= prefixSum.size()) {
                ans[i] = -1;
                break;
            } else {
                ans[i] = max(ans[i], right + 1);
            }
        }
    }
    return ans;
}


bool Check(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}
vector<vector<int>> Flip(vector<vector<int>>& grid, int row, int col)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int direction[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    vector<vector<int>> ans(m, vector<int>(n, 0));
    ans = grid;

    ans[row][col] = 1 - grid[row][col];
    for (i = 0; i < 4; i++) {
        int nr = row + direction[i][0];
        int nc = col + direction[i][1];

        if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
            ans[nr][nc] = 1 - grid[nr][nc];
        }
    }
    return ans;
}
vector<int> TwoDimonsionTolinear(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    vector<int> ans;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            ans.emplace_back(grid[i][j]);
        }
    }
    return ans;
}
int minFlips(vector<vector<int>>& mat)
{
    int i, j, k;
    int m = mat.size();
    int n = mat[0].size();
    int step = 0;
    set<vector<int>> data;
    queue<vector<vector<int>>> q;

    if (Check(mat)) {
        return step;
    }

    q.push(mat);
    data.emplace(TwoDimonsionTolinear(mat));
    while (q.size()) {
        int size = q.size();
        for (k = 0; k < size; k++) {
            vector<vector<int>> t = q.front();
            q.pop();

            if (Check(t)) {
                return step;
            }
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    vector<vector<int>> vv = Flip(t, i, j);
                    vector<int> v = TwoDimonsionTolinear(vv);
                    if (data.find(v) == data.end()) {
                        data.emplace(v);
                        q.push(vv);
                    }
                }
            }
        }
        step++;
    }
    return -1;
}


int CountPlusSign(vector<vector<int>>& grid, int row, int col)
{
    if (grid[row][col] == 0) {
        return 0;
    }

    int cnt;
    int m, n, a, b, c, d;

    m = grid.size();
    n = grid[0].size();

    cnt = 1;
    a = b = row;
    c = d = col;
    while (1) {
        a--;
        b++;
        c--;
        d++;

        if (a < 0 || b >= m || c < 0 || d >= n) {
            break;
        }
        if (grid[a][col] == 0 || grid[b][col] == 0 || grid[row][c] == 0 || grid[row][d] == 0) {
            break;
        }
        cnt++;
    }
    return cnt;
}
int orderOfLargestPlusSign(int n, vector<vector<int>>& mines)
{
    int i, j;
    int ans;

    vector<vector<int>> grid(n, vector<int>(n, 1));
    for (auto v : mines) {
        grid[v[0]][v[1]] = 0;
    }
    ans = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            ans = max(ans, CountPlusSign(grid, i ,j));
        }
    }
    return ans;
}


long long maximumSubarraySum(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int t;
    long long ans;
    long long sum;
    unordered_map<int, int> numsFreq;
    map<int, unordered_set<int>, greater<int>> freqOfNums;

    ans = sum = 0;
    for (i = 0; i < k; i++) {
        sum += nums[i];
        if (numsFreq.find(nums[i]) == numsFreq.end()) {
            freqOfNums[1].emplace(nums[i]);
        } else {
            t = numsFreq[nums[i]];
            freqOfNums[t].erase(nums[i]);
            freqOfNums[t + 1].emplace(nums[i]);
        }
        numsFreq[nums[i]]++;
    }
    if (freqOfNums.begin()->first == 1) {
        ans = max(ans, sum);
    }
    for (i = 1; i <= n - k; i++) {
        t = numsFreq[nums[i - 1]];
        if (t == 1) {
            numsFreq.erase(nums[i - 1]);
            if (freqOfNums[1].size() == 1) {
                freqOfNums.erase(1);
            } else {
                freqOfNums[1].erase(nums[i - 1]);
            }
        } else {
            numsFreq[nums[i - 1]]--;
            if (freqOfNums[t].size() == 1) {
                freqOfNums.erase(t);
            } else {
                freqOfNums[t].erase(nums[i - 1]);
            }
            freqOfNums[t - 1].emplace(nums[i - 1]);
        }
        sum -= nums[i - 1];

        if (numsFreq.find(nums[i + k - 1]) == numsFreq.end()) {
            freqOfNums[1].emplace(nums[i + k - 1]);
        } else {
            t = numsFreq[nums[i + k - 1]];
            freqOfNums[t].erase(nums[i + k - 1]);
            freqOfNums[t + 1].emplace(nums[i + k - 1]);
        }
        numsFreq[nums[i + k - 1]]++;
        sum += nums[i + k - 1];

        if (freqOfNums.begin()->first == 1) {
            ans = max(ans, sum);
        }
    }
    return ans;
}


int FindIdx(string& s, vector<int>& v, int idx, string &ch) // 找到v里第一个大于等于idx的值
{
    int left, right, mid;

    left = 0;
    right = v.size() - 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        if (v[mid] >= idx) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    // printf ("s = %s, left = %d, right = %d\n", ch.c_str(), left, right);
    if (left >= v.size() || (right == -1 && v[right + 1] < idx)) {
        return -1;
    }
    return v[right + 1];
}
int numMatchingSubseq(string s, vector<string>& words)
{
    int i, j;
    int n = s.size();
    int cnt, t;
    unordered_map<char, vector<int>> strInfo;

    for (i = 0; i < n; i++) {
        strInfo[s[i]].emplace_back(i);
    }
    cnt = 0;
    for (i = 0; i < words.size(); i++) {
        t = -1;
        for (j = 0; j < words[i].size(); j++) {
            if (strInfo.find(words[i][j]) == strInfo.end()) {
                break;
            }
            t = FindIdx(s, strInfo[words[i][j]], t + 1, words[i]);
            // printf ("%s : t = %d\n", words[i].c_str(), t);
            if (t == -1) {
                break;
            }
        }
        if (j == words[i].size()) {
            cnt++;
        }
    }
    return cnt;
}


long long minimumFuelCost(vector<vector<int>>& roads, int seats)
{
    unordered_map<int, vector<int>> routes;
    unordered_map<int, int> dist;
    unordered_set<int> city;
    unordered_set<int> visited;
    queue<int> q;
    int i, j;
    int n = roads.size();
    int m, t;

    if (n == 0) {
        return 0;
    }
    for (auto r : roads) {
        routes[r[0]].emplace_back(r[1]);
        routes[r[1]].emplace_back(r[0]);
        city.emplace(r[0]);
        city.emplace(r[1]);
    }

    q.push(0);
    while (q.size()) {
        m = q.size();
        for (i = 0; i < m; i++) {
            t = q.front();
            q.pop();
            cout << t << endl;
            if (visited.find(t) != visited.end()) {
                continue;
            }
            visited.emplace(t);
            for (j = 0; j < routes[t].size(); j++) {
                //if (visited.find(routes[i][j]) == visited.end()) {
                    q.push(routes[i][j]);
                    dist[routes[i][j]]++;
                //}
            }
        }
    }
    for (auto it : dist) {
        printf ("%d : %d\n", it.first, it.second);
    }
    return 0;
}


vector<int> eventualSafeNodes(vector<vector<int>>& graph)
{
    int i;
    int n = graph.size();
    vector<int> terminalNode;
    vector<int> safeNode;
    for (i = 0; i < n; i++) {
        if (graph[i].size() == 0) {
            terminalNode.emplace_back(i);
        }
    }


    return safeNode;
}



int expressiveWords(string s, vector<string>& words)
{
    int i, j;
    int n, cnt, ans;
    int m = s.size();
    int nums = words.size();
    bool flag = false;
    vector<pair<char, int>> ss, ww;

    cnt = 1;
    for (i = 1; i < m; i++) {
        if (s[i] == s[i - 1]) {
            cnt++;
        } else {
            ss.emplace_back(make_pair(s[i - 1], cnt));
            cnt = 1;
        }
    }
    ss.emplace_back(make_pair(s[m - 1], cnt));

    ans = 0;
    for (i = 0; i < nums; i++) {
        n = words[i].size();
        if (n > m) {
            continue;
        }

        cnt = 1;
        ww.clear();
        for (j = 1; j < n; j++) {
            if (words[i][j] == words[i][j - 1]) {
                cnt++;
            } else {
                ww.emplace_back(make_pair(words[i][j - 1], cnt));
                cnt = 1;
            }
        }
        ww.emplace_back(make_pair(words[i][n - 1], cnt));

        if (ss.size() != ww.size()) {
            continue;
        }
        flag = false;
        for (j = 0; j < ss.size(); j++) {
            if (ss[j].first != ww[j].first || ww[j].second > ss[j].second) {
                break;
            }
            if (ww[j].second != ss[j].second && ss[j].second < 3) {
                break;
            }
            if (ww[j].second + 2 <= ss[j].second) {
                flag = true;
            }
        }
        if (j != ss.size()) {
            continue;
        }
        if (flag) {
            ans++;
        }
    }
    return ans;
}


bool isPossible(int n, vector<vector<int>>& edges)
{
    unordered_map<int, int> nodeDegree;
    unordered_map<int, unordered_set<int>> ed;
    unordered_map<int, unordered_set<int>> oddDegreeEdge;
    for (auto e : edges) {
        nodeDegree[e[0]]++;
        nodeDegree[e[1]]++;
        ed[e[0]].emplace(e[1]);
        ed[e[1]].emplace(e[0]);
    }
    vector<int> oddDegreeNodes;
    for (auto it : nodeDegree) {
        if (it.second % 2 == 1) {
            oddDegreeEdge[it.first] = ed[it.first];
            oddDegreeNodes.emplace_back(it.first);
        }
    }
    int m = oddDegreeNodes.size();
    if (m != 0 && m != 2 && m != 4) {
        return false;
    }

    /* cout << "cnt = " << m << endl;
    for (auto it : oddDegreeEdge) {
        printf ("%d: ", it.first);
        for (auto it1 : it.second) {
            printf ("%d ", it1);
        }
        printf ("\n");
    }
    for (auto nn : oddDegreeNodes) cout << nn << " "; cout << endl; */

    if (m == 2) {
        for (int i = 1; i <= n; i++) {
            if (oddDegreeEdge[oddDegreeNodes[0]].count(i) == 0 &&
                oddDegreeEdge[oddDegreeNodes[1]].count(i) == 0) {
                return true;
            }
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[1]) == 1) {
            return false;
        }
    }
    if (m == 4) {
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[1]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[2]].count(oddDegreeNodes[3]) == 0) {
            return true;
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[2]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[1]].count(oddDegreeNodes[3]) == 0) {
            return true;
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[3]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[1]].count(oddDegreeNodes[2]) == 0) {
            return true;
        }
        return false;
    }
    return true;
}


vector<vector<int>> SplitVector(vector<int>& nums)
{
    int i;
    int a, b;
    int n = nums.size();
    vector<int> v;
    vector<vector<int>> ans;

    int cnt = 0;
    for (i = 0; i < n; i++) {
        if (nums[i] == 0) {
            cnt++;
            if (cnt == 1) {
                a = i;
            } else {
                b = i;
                if (b - a > 1) {
                    ans.emplace_back(v);
                    v.clear();
                }
                cnt = 1;
                a = i;
            }
        } else {
            v.emplace_back(nums[i] > 0 ? 1 : -1);
        }
    }
    return ans;
}
int getMaxLen(vector<int>& nums)
{
    int i;
    int m, t;
    int n = nums.size();
    vector<int> v(n + 2, 0);

    for (i = 0; i < n; i++) {
        v[i + 1] = nums[i];
    }

    int ans = 0;
    vector<int> negIdx;
    vector<vector<int>> arrays = SplitVector(v);
    for (auto arr : arrays) {
        n = arr.size();
        negIdx.clear();
        for (i = 0; i < n; i++) {
            if (arr[i] < 0) {
                negIdx.emplace_back(i);
            }
        }
        m = negIdx.size();
        if (m % 2 == 0) {
            ans = max(ans, n);
            continue;
        }
        t = max(n - (negIdx[0] + 1), negIdx[m - 1]);
        ans = max(ans, t);
    }
    return ans;
}

bool isItPossible(string word1, string word2)
{
    unordered_map<int, int> dict1, dict2;
    for (auto w : word1) {
        dict1[w]++;
    }
    for (auto w : word2) {
        dict2[w]++;
    }
    int m = dict1.size();
    int n = dict2.size();
    if (abs(m - n) > 2) {
        return false;
    }
    if (m == n) {
        for (auto it1 : dict1) {
            if (dict2.find(it1.first) != dict2.end()) {
                return true;
            } else {
                for (auto it2 : dict2) {
                    if (dict1.find(it2.first) == dict1.end()) {
                        if (it1.second == it2.second || (it1.second > 1 && it2.second > 1)) {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    if (abs(m - n) == 1) {
        if (m > n) {
            for (auto it1 : dict1) {
                if (dict2.find(it1.first) == dict2.end() && it1.second > 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end() && it2.second > 1) {
                            return true;
                        }
                    }
                }
                if (dict2.find(it1.first) == dict2.end() && it1.second == 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end() && it1.second == it2.second) {
                            return true;
                        }
                    }
                }
            }
        } else {
            for (auto it2 : dict2) {
                if (dict1.find(it2.first) == dict1.end() && it2.second > 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end() && it1.second > 1) {
                            return true;
                        }
                    }
                }
                if (dict1.find(it2.first) == dict1.end() && it2.second == 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end() && it1.second == it2.second) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    if (abs(m - n) == 2) {
        if (m > n) {
            for (auto it1 : dict1) {
                if (dict2.find(it1.first) == dict2.end() && it1.second == 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end()) {
                            return true;
                        }
                    }
                }
            }
        } else {
            for (auto it2 : dict2) {
                if (dict1.find(it2.first) == dict1.end() && it2.second == 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end()) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool isItPossible_1(string word1, string word2)
{
    unordered_map<char, int> dict1, dict2, tmp1, tmp2;
    for (auto w : word1) {
        dict1[w]++;
    }
    for (auto w : word2) {
        dict2[w]++;
    }
    int m = dict1.size();
    int n = dict2.size();
    if (abs(m - n) > 2) {
        return false;
    }
    tmp1 = dict1;
    tmp2 = dict2;
    for (auto it1 : dict1) {
        if (it1.second == 1) {
            tmp1.erase(it1.first);
        } else {
            tmp1[it1.first]--;
        }
        for (auto it2 : dict2) {
            if (it2.second == 1) {
                tmp2.erase(it2.first);
            } else {
                tmp2[it2.first]--;
            }
            tmp2[it1.first]++;
            tmp1[it2.first]++;
            if (tmp1.size() == tmp2.size()) {
                return true;
            }
            // 还原
            if (tmp1[it2.first] == 1) {
                tmp1.erase(it2.first);
            } else {
                tmp1[it2.first]--;
            }
            tmp2[it2.first]++;
            if (tmp2[it1.first] == 1) {
                tmp2.erase(it1.first);
            } else {
                tmp2[it1.first]--;
            }
        }
        // 还原
        tmp1[it1.first]++;
    }
    return false;
}

bool isItPossible_2(string word1, string word2)
{
    int i, j, k;
    int a[26] = {0};
    int b[26] = {0};
    int cntA, cntB;

    for (auto w : word1) {
        a[w - 'a']++;
    }
    for (auto w : word2) {
        b[w - 'a']++;
    }
    for (i = 0; i < 26; i++) {
        if (a[i] > 0) {
            a[i]--;
            b[i]++;
            for (j = 0; j < 26; j++) {
                if (b[j] == 0) {
                    continue;
                }
                if ((i == j && b[j] == 1)) {
                    continue;
                }
                if (b[j] > 0) {
                    b[j]--;
                    a[j]++;
                }
                cntA = cntB = 0;
                for (k = 0; k < 26; k++) {
                    if (a[k] > 0) {
                        cntA++;
                    }
                    if (b[k] > 0) {
                        cntB++;
                    }
                }
                if (cntA == cntB) {
                    return true;
                }
                // 还原
                b[j]++;
                a[j]--;
            }
            // 还原
            a[i]++;
            b[i]--;
        }
    }
    return false;
}



map<string, string> dict;
map<string, set<string>> prefixDict;
map<string, int> prefixFreq;
void GenerateUniqueString(int n, int idx, int k, string t)
{
    if (idx == n) {
        string prefix = t.substr(0, t.size() - 1);
        string suffix = t.substr(1);
        dict[t] = suffix;
        prefixFreq[prefix]++;
        prefixDict[prefix].emplace(t);
        return;
    }
    int i;
    for (i = 0; i < k; i++) {
        t[idx] = i + '0';
        GenerateUniqueString(n, idx + 1, k, t);
    }
}
string crackSafe(int n, int k)
{
    int i;
    int cnt;
    string ans;
    if (n == 1) {
        for (i = 0; i < k; i++) {
            ans += i + '0';
        }
        return ans;
    }
    string t(n, ' ');
    GenerateUniqueString(n, 0, k, t);

    /*priority_queue<pair<int, string>, vector<pair<int, string>>> pq;
    for (auto pf : prefixFreq) {
        pq.push(make_pair(pf.second, pf.first));
    }*/
    ans = dict.begin()->first;
    string curString = ans;
    string curPrefix = ans.substr(0, n - 1);
    string nextPrefix = dict.begin()->second;
    prefixFreq[curPrefix]--;
    prefixDict[curPrefix].erase(curString);

    string tmp;
    while (1) {

        cnt = 0;
        for (auto it : prefixDict[nextPrefix]) {
            nextPrefix = dict[it];
            if (prefixFreq[nextPrefix] > cnt) {
                cnt = prefixFreq[nextPrefix];
                curString = it;
            }
        }
        curPrefix = curString.substr(0, curString.size() - 1);
        ans += curString[curString.size() - 1];
        nextPrefix = dict[curString];

        if (prefixFreq[curPrefix] == 1) {
            prefixFreq.erase(curPrefix);
        } else {
            prefixFreq[curPrefix]--;
        }

        if (prefixDict[curPrefix].size() == 1) {
            prefixDict.erase(curPrefix);
        } else {
            prefixDict[curPrefix].erase(curString);
        }
        if (prefixDict.size() == 0) {
            break;
        }
    }
    return ans;
}


void DeleteTree(TreeNode* &node)
{
    if (node == nullptr) {
        return;
    }
    DeleteTree(node->left);
    DeleteTree(node->right);
    delete node;
    node = nullptr;
}
void DFSCheckTreeSum(TreeNode* &node, int curSum, int limit, bool& flag)
{
    if (node == nullptr) {
        return;
    }
    curSum += node->val;
    if (node->left == nullptr && node->right == nullptr) {
        if (curSum >= limit) {
            flag = false;
        }
        return;
    }
    DFSCheckTreeSum(node->left, curSum, limit, flag);
    DFSCheckTreeSum(node->right, curSum, limit, flag);
}
void DFSSCanTree(TreeNode* &node, TreeNode *parent, char parentBranch, int curSum, int limit)
{
    bool flag = true;
    if (node == nullptr) {
        return;
    }
    DFSCheckTreeSum(node, curSum, limit, flag);
    if (flag) {
        DeleteTree(node);
        flag = true;
        if (parent != nullptr) {
            if (parentBranch == 'L') {
                parent->left = nullptr;
            } else {
                parent->right = nullptr;
            }
        } else {

        }
    }
    if (node != nullptr) {
        DFSSCanTree(node->left, node, 'L', curSum + node->val, limit);
        DFSSCanTree(node->right, node, 'R', curSum + node->val, limit);
    }
}
TreeNode* sufficientSubset(TreeNode* root, int limit)
{
    DFSSCanTree(root, nullptr, 'L', 0, limit);
    return root;
}


int waysToMakeFair(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<int> oddSum, evenSum;

    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        if (i % 2 == 0) {
            if (evenSum.empty()) {
                evenSum.emplace_back(nums[i]);
            } else {
                evenSum.emplace_back(evenSum[evenSum.size() - 1] + nums[i]);
            }
        } else {
            if (oddSum.empty()) {
                oddSum.emplace_back(nums[i]);
            } else {
                oddSum.emplace_back(oddSum[oddSum.size() - 1] + nums[i]);
            }
        }
    }
    int ans = 0;
    int odds = 0;
    int evens = 0;
    int p = oddSum.size();
    int q = evenSum.size();
    for (i = 0; i < n; i++) {
       if (i % 2 == 1) {
            if (i / 2 > 0) {
                odds = oddSum[i / 2 - 1] + evenSum[q - 1] - evenSum[i / 2];
            } else {
                odds = evenSum[q - 1] - evenSum[i / 2];
            }
            evens = evenSum[i / 2] + oddSum[p - 1] - oddSum[i / 2];
        } else {
            if (i == 0) {
                odds = evenSum[q - 1] - evenSum[0];
                evens = oddSum[p - 1];
            } else {
                odds = oddSum[i / 2 - 1] + evenSum[q - 1] - evenSum[i / 2];
                evens = evenSum[i / 2 - 1] + oddSum[p - 1] - oddSum[i / 2 - 1];
            }
        }
        if (odds == evens) {
            ans++;
        }
    }
    return ans;
}

static bool MaxScoreCmp(const pair<int, int>& a, const pair<int, int>& b)
{
    if (a.first == b.first) {
        return a.second > b.second;
    }
    return a.first > b.first;
}
long long maxScore(vector<int>& nums1, vector<int>& nums2, int k)
{
    int i;
    int n = nums1.size();
    int t;
    long long ans, sum;
    priority_queue<int, vector<int>, greater<>> pq;
    vector<pair<int, int>> vp;

    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(nums2[i], i));
    }
    sort(vp.begin(), vp.end(), MaxScoreCmp);
    sum = 0;
    for (i = 0; i < k; i++) {
        t = nums1[vp[i].second];
        sum += t;
        pq.push(t);
    }
    ans = sum * vp[k - 1].first;
    for (i = k; i < n; i++) {
        t = nums1[vp[i].second];
        if (pq.top() < t) {
            sum += t - pq.top();
            pq.pop();
            pq.push(t);
            ans = max(ans, sum * vp[i].first);
        }
    }
    return ans;
}


int FastPow(int a, int b)
{
    int ans = 1, base = a;
    while (b != 0) {
        if ((b & 1) != 0)
            ans = (static_cast<long long>(ans) * base) % mod;
        base = (static_cast<long long>(base) * base) % mod;
        b >>= 1;
    }
    return ans;
}


long long putMarbles(vector<int>& weights, int k)
{
    int i;
    int n = weights.size();
    long long ans;
    vector<int> val(n - 1);

    for (i = 1; i < n; i++) {
        val[i - 1] = weights[i - 1] + weights[i];
    }
    sort(val.begin(), val.end());

    ans = 0;
    for (i = 0; i < k - 1; i++) {
        ans += val[n - 2 - i] - val[i];
    }
    return ans;
}


ListNode* mergeInBetween(ListNode* list1, int a, int b, ListNode* list2)
{
    int cnt;
    ListNode *prev = nullptr;
    ListNode *cur = list2;
    while (cur->next) {
        cur = cur->next;
    }
    ListNode *list2Tail = cur;

    cur = list1;
    cnt = 0;
    while (cur) {
        if (cnt == a) {
            prev->next = list2;
        }
        if (cnt == b) {
            list2Tail->next = cur->next;
            break;
        }
        cnt++;
        prev = cur;
        cur = cur->next;
    }
    return list1;
}


static bool longestStrChainCMP(const string& a, const string& b)
{
    if (a.size() != b.size()) {
        return a.size() < b.size();
    }
    return a < b;
}
bool IsParent(string& a, string& b) // a - cur  b - parent
{
    int m = a.size();
    int n = b.size();
    int i, j;

    if (n + 1 != m) {
        return false;
    }
    i = j = 0;
    while (j < n && i < m) {
        if (a[i] == b[j]) {
            i++;
            j++;
        } else {
            i++;
        }
    }
    if (j == n) {
        return true;
    }
    return false;
} 
int longestStrChain(vector<string>& words)
{
    int i, j;
    int n = words.size();
    vector<int> dp(n, 1);
    sort(words.begin(), words.end(), longestStrChainCMP);
    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (IsParent(words[i], words[j])) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    int ans = 0;
    for (auto d : dp) {
        ans = max(ans, d); 
    }
    return ans;
}


unordered_map<TreeNode *, unordered_set<TreeNode *>> edges;
vector<TreeNode *> leaves;
void DFSCreateGrid(TreeNode * node, TreeNode *parent)
{
    if (node == nullptr) {
        return;
    }
    if (node->left == nullptr && node->right == nullptr) {
        leaves.emplace_back(node);
    }
    if (parent != nullptr) {
        edges[node].emplace(parent);
        edges[parent].emplace(node);
    }
    DFSCreateGrid(node->left, node);
    DFSCreateGrid(node->right, node);
}
int CheckCanReach(TreeNode *a, int distance)
{
    int i, n;
    int step, ans;
    queue<TreeNode *> q;
    unordered_set<TreeNode *> visited;

    q.push(a);
    step = 0;
    ans = 0;
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            TreeNode *t = q.front();
            q.pop();
            if (t != a && t->left == nullptr && t->right == nullptr) {
                ans++;
            }
            visited.emplace(t);
            for (auto it : edges[t]) {
                if (visited.count(it) == 1) {
                    continue;
                }
                q.push(it);
            }
        }
        step++;
        if (step > distance) {
            break;
        }
    }
    return ans;
}
int countPairs(TreeNode* root, int distance)
{
    int ans;
    DFSCreateGrid(root, nullptr);

    int num = leaves.size();
    if (num < 2) {
        return 0;
    }
    int i, j;
    ans = 0;
    for (i = 0; i < num; i++) {
        ans += CheckCanReach(leaves[i], distance);
    }
    return ans / 2;
}


void DSFAdjustRoute(int cur, int parent, unordered_map<int, vector<int>>& edges, set<pair<int, int>>& realRoutePair, 
    unordered_set<int>& visited, int& ans)
{
    visited.emplace(cur);
    if (parent != -1) {
        if (realRoutePair.count(make_pair(cur, parent)) == 0) {
            ans++;
        }
    }
    for (auto it : edges[cur]) {
        if (visited.find(it) != visited.end()) {
            continue;
        }
        DSFAdjustRoute(it, cur, edges, realRoutePair, visited, ans);
    }
}
int minReorder(int n, vector<vector<int>>& connections)
{
    int ans;
    unordered_map<int, vector<int>> edges;
    set<pair<int, int>> realRoutePair;
    unordered_set<int> visited;
    for (auto conn : connections) {
        edges[conn[0]].emplace_back(conn[1]);
        edges[conn[1]].emplace_back(conn[0]);
        realRoutePair.emplace(make_pair(conn[0], conn[1]));
    }

    ans = 0;
    DSFAdjustRoute(0, -1, edges, realRoutePair, visited, ans);
    return ans;
}


string findContestMatch(int n)
{
    int i, j;
    string ans;
    vector<string> v, t;

    for (i = 1; i <= n; i++) {
        v.emplace_back(to_string(i));
    }
    while (n != 1) {
        i = 0;
        j = v.size() - 1;
        t.clear();
        while (i < j) {
            t.emplace_back("(" + v[i] + "," + v[j] + ")");
            i++;
            j--;
        }
        v = t;
        n /= 2;
    }
    ans = v[0];
    return ans;
}


void CreateWordTrie(Trie<char> *root, string& word)
{
    int i, j;
    int n, m;

    n = word.size();
    for (i = 0; i < n; i++) {
        if (root->children.size() == 0) {
            Trie<char> *node = new Trie(word[i]);
            root->children.emplace_back(node);
            root = node;
        } else {
            m = root->children.size();
            for (j = 0; j < m; j++) {
                if (root->children[j]->val == word[i]) {
                    root = root->children[j];
                    break;
                }
            }
            if (j == m) {
                Trie<char> *node = new Trie(word[i]);
                root->children.emplace_back(node);
                root = node;
            }
        }
    }
    root->IsEnd = true;
}
void DFSScanWordTrie(Trie<char> *root, string& t, int point, unordered_map<string, int>& wordPrefixPoint)
{
    int i;
    t.push_back(root->val);
    if (root->IsEnd) {
        point++;
        wordPrefixPoint[t.substr(1)] = point; // 去掉字符串首位的'/'
    }
    for (i = 0; i < root->children.size(); i++) {
        DFSScanWordTrie(root->children[i], t, point, wordPrefixPoint);
        t.pop_back();
    }
}
string longestWord(vector<string>& words)
{
    int i, j;
    int m = words.size();
    int n;
    Trie<char> *root = new Trie('/');

    for (i = 0; i < m; i++) {
        CreateWordTrie(root, words[i]);
    }

    unordered_map<string, int> wordPrefixPoint;
    string t;
    DFSScanWordTrie(root, t, 0, wordPrefixPoint);

    string ans = "";
    for (auto it : wordPrefixPoint) {
        // cout << it.first << " " << it.second << endl;
        if (it.first.size() == it.second) {
            if (it.second > ans.size()) {
                ans = it.first;
            } else if (it.second == ans.size()) {
                ans = min(ans, it.first);
            }
        }
    }
    return ans;
}


int maxSumAfterOperation(vector<int>& nums)
{
    int ans;
    int n = nums.size();
    int i;
    int a, b;
    vector<vector<int>> dp(n, vector<int>(2, 0));
    // 最大子数组的变种 dp[i][0] -- 以nums[i]结尾的最大子数组和,不使用平方; dp[i][1] -- 以nums[i]结尾的最大子数组和,使用平方
    dp[0][0] = nums[0];
    dp[0][1] = nums[0] * nums[0];
    ans = max(dp[0][0], dp[0][1]);
    for (i = 1; i < n; i++) {
        dp[i][0] = dp[i - 1][0] > 0 ? dp[i - 1][0] + nums[i] : nums[i];
        a = dp[i - 1][0] > 0 ? dp[i - 1][0] + nums[i] * nums[i] : nums[i] * nums[i];
        b = dp[i - 1][1] > 0 ? dp[i - 1][1] + nums[i] : nums[i];
        dp[i][1] = max(a, b);
        ans = max(ans, max(dp[i][0], dp[i][1]));
    }
    return ans;
}


bool canBeValid(string s, string locked)
{
    int i;
    int n = s.size();
    stack<int> lockedIdxStack;
    stack<int> freeIdxStack;

    if (n % 2 == 1 || (s[0] == ')' && locked[0] == '1') || (s[n - 1] == '(' && locked[n - 1] == '1')) {
        return false;
    }

    for (i = 0; i < n; i++) {
        if (locked[i] == '0') {
            freeIdxStack.push(i);
            continue;
        }
        if (s[i] == '(') {
            lockedIdxStack.push(i);
        } else {
            if (lockedIdxStack.empty()) {
                if (freeIdxStack.empty()) {
                    return false;
                } else {
                    freeIdxStack.pop();
                }
            } else {
                lockedIdxStack.pop();
            }
        }
    }
    while (lockedIdxStack.size()) {
        if (freeIdxStack.empty() || lockedIdxStack.top() > freeIdxStack.top()) {
            return false;
        }
        lockedIdxStack.pop();
        freeIdxStack.pop();
    }
    return lockedIdxStack.empty();
}


bool TrySessions(vector<int>& tasks, int sessionTime, int bucketSize)
{
    int i;
    int n = tasks.size();
    bool isOk = false;
    vector<int> bucket(bucketSize, 0);
    CheckBucket(tasks, sessionTime, bucket, 0, isOk);
    return isOk;
}
void CheckBucket(vector<int>& tasks, int sessionTime, vector<int> &bucket, int idx, bool& isOk)
{
    int i;
    int n = bucket.size();

    if (isOk) {
        return;
    }
    if (idx == tasks.size()) {
        isOk = true;
        return;
    }
    for (i = 0; i < min(n, idx + 1); i++) {
        if (bucket[i] + tasks[idx] <= sessionTime) {
            bucket[i] += tasks[idx];
            CheckBucket(tasks, sessionTime, bucket, idx + 1, isOk);
            bucket[i] -= tasks[idx];
        }
    }
}
int minSessions(vector<int>& tasks, int sessionTime)
{
    int bucketSize;
    int n = tasks.size();
    int ans;
    sort (tasks.rbegin(), tasks.rend());
    ans = n;
    for (bucketSize = 1; bucketSize <= n; bucketSize++) {
        if (TrySessions(tasks, sessionTime, bucketSize)) {
            ans = bucketSize;
            break;
        }
    }
    return ans;
}


void DFSTrySum(vector<int>& nums, int curSum, int curMatchSubArr, int val, int k, vector<bool>& visited, bool& matchAll, int idx)
{
    int i;
    if (matchAll == true) {
        return;
    }
    for (i = idx; i < nums.size(); i++) {
        if (visited[i] == false) {
            if (curSum - curMatchSubArr * val + nums[i] == val) {
                if (curMatchSubArr + 1 == k) {
                    matchAll = true;
                    return;
                } else {
                    visited[i] = true;
                    DFSTrySum(nums, curSum + nums[i], curMatchSubArr + 1, val, k, visited, matchAll, 0);
                    visited[i] = false;
                }
            } else if (curSum - curMatchSubArr * val + nums[i] < val) {
                visited[i] = true;
                DFSTrySum(nums, curSum + nums[i], curMatchSubArr, val, k, visited, matchAll, i + 1);
                visited[i] = false;
            }
        }
    }
}
bool makesquare(vector<int>& matchsticks)
{
    int sum = 0;
    int n = matchsticks.size();

    if (n < 4) {
        return false;
    }
    for (auto m : matchsticks) {
        sum += m;
    }
    if (sum % 4 != 0) {
        return false;
    }
    int edgeLen = sum / 4;
    sort(matchsticks.rbegin(), matchsticks.rend());
    if (matchsticks[0] > edgeLen) {
        return false;
    }
    vector<int> v;
    bool f = false;
    vector<bool> visited(n, false);
    DFSTrySum(matchsticks, 0, 0, edgeLen, 4, visited, f, 0);
    return f;
}


// 从数组取k个互不相邻元素和最小值
// dp[i][j] 从前i个物品选j个物品的最小和
// dp[i][j] = min(dp[i - 1][j], dp[i - 2][j - 1] + nums[i])
int minGetSum(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();

    if (n == 1) {
        return nums[0];
    } else if (n == 2) {
        return min(nums[0], nums[1]);
    }
    vector<vector<long long>> dp(n, vector<long long>(k + 1, 0x3f3f3f3f));
    dp[0][0] = 0;
    dp[0][1] = nums[0];
    dp[1][1] = min(nums[0], nums[1]);
    for (i = 2; i < n; i++) {
        for (j = 1; j <= k; j++) {
            // dp[i][j] = min(dp[i - 1][j], dp[i - 2][j - 1] + nums[i]);
            if (dp[i - 2][j - 1] + nums[i] < dp[i - 1][j]) {
                dp[i][j] = dp[i - 2][j - 1] + nums[i];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[n - 1][k];
}


long long minCost(vector<int>& basket1, vector<int>& basket2)
{
    int i;
    int k;
    int n = basket1.size();
    long long ans;
    map<int, int> m1, m2;
    for (i = 0; i < n; i++) {
        m1[basket1[i]]++;
        m2[basket2[i]]++;
    }
    vector<vector<int>> v1, v2;
    for (auto it : m1) {
        if (m2.find(it.first) == m2.end()) {
            if (it.second % 2 != 0) {
                return -1;
            }
            v1.push_back({it.first, it.second / 2});
        } else {
            if (abs(it.second - m2[it.first]) % 2 != 0) {
                return -1;
            }
            if (it.second > m2[it.first]) {
                v1.push_back({it.first, (it.second - m2[it.first]) / 2});
            } else if (it.second < m2[it.first]) {
                v2.push_back({it.first, (m2[it.first] - it.second) / 2});
            }
        }
    }
    for (auto it : m2) {
        if (m1.find(it.first) == m1.end()) {
            if (it.second % 2 != 0) {
                return -1;
            }
            v2.push_back({it.first, it.second / 2});
        }
    }
    /* for (auto v : v1) cout << v[0] << ":" << v[1] << endl;
    cout << endl;
    for (auto v : v2) cout << v[0] << ":" << v[1] << endl;
    cout << endl;
    */
    ans = 0;
    k = min(m1.begin()->first, m2.begin()->first) * 2;
    vector<int> vv;
    for (auto v : v1) {
        for (i = 0; i < v[1]; i++) {
            vv.emplace_back(v[0]);
        }
    }
    for (auto v : v2) {
        for (i = 0; i < v[1]; i++) {
            vv.emplace_back(v[0]);
        }
    }

    sort(vv.begin(), vv.end());
    n = vv.size();
    for (i = 0; i < n / 2; i++) {
        ans += min(k, vv[i]);
    }
    return ans;
}


int maxCount(vector<int>& banned, int n, long long maxSum)
{
    int i;
    int banSize;
    int start, end;
    int cnt;
    int left, right, mid;
    long long t;
    vector<int> ban;

    sort(banned.begin(), banned.end());
    ban.emplace_back(0);
    ban.insert(ban.end(), banned.begin(), banned.end());
    ban.emplace_back(n + 1);
    banSize = ban.size();

    cnt = 0;
    for (i = 1; i < banSize; i++) {
        start = ban[i - 1] + 1;
        end = ban[i] - 1;
        if (start > end) {
            continue;
        }
        t = static_cast<long long>(start + end) * (end - start + 1) / 2;
        maxSum -= t;
        if (maxSum == 0) {
            cnt += end - start + 1;
            break;
        } else if (maxSum > 0) {
            cnt += end - start + 1;
            continue;
        }
        left = start;
        right = end;
        maxSum += t; // 还原
        while (left <= right) { // 所求为right
            mid = (right - left) / 2 + left;
            t = static_cast<long long>(start + mid) * (mid - start + 1) / 2;
            if (t <= maxSum) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        // cout << left << " " << right << endl;
        cnt += right - start + 1;
        break;
    }
    return cnt;
}


bool CheckContinuousArr(int lastArrVal, vector<int>& arr)
{
    int i;

    sort(arr.begin(), arr.end());
    if (lastArrVal + 1 != arr[0]) {
        return false;
    }
    i = 1;
    for (; i < arr.size(); i++) {
        if (arr[i - 1] + 1 != arr[i]) {
            break;
        }
    }
    if (i != arr.size()) {
        return false;
    }
    return true;
} 
void TrySplit(vector<int>& arr, int left, vector<int>& record, int cnt, int& ans, bool& find)
{
    int i, j;
    int n = arr.size();
    if (find) {
        return;
    }
    if (left == n) {
        find = true;
        ans = cnt;
        return;
    }
    for (i = left; i < n; i++) {
        if (record.size() == 0) {
            record.assign(arr.begin() + left, arr.begin() + i + 1);
            if (CheckContinuousArr(-1, record) == false) {
                record.clear();
                continue;
            }
            TrySplit(arr, i + 1, record, cnt + 1, ans, find);
        } else {
            vector<int> v;
            v.assign(arr.begin() + left, arr.begin() + i + 1);
            if (CheckContinuousArr(record[record.size() - 1], v) == false) {
                v.clear();
                continue;
            }
            record.insert(record.end(), v.begin(), v.end());
            TrySplit(arr, i + 1, record, cnt + 1, ans, find);
        }
    }
}
int maxChunksToSorted(vector<int>& arr)
{
    int ans = 0;
    bool find = false;
    vector<int> record;
    TrySplit(arr, 0, record, 0, ans, find);
    return ans;
}



vector<string> removeSubfolders(vector<string>& folder)
{
    int i, j;
    int n = folder.size();

    sort (folder.begin(), folder.end());
    vector<string> ans;
    ans.emplace_back(folder[0]);
    for (i = 0; i < n;) {
        for (j = i + 1; j < n; j++) {
            if (strncmp((folder[i] + "/").c_str(), (folder[j] + "/").c_str(), folder[i].size() + 1) != 0) {
                ans.emplace_back(folder[j]);
                break;
            }
        }
        i = j;
        if (j == n) {
            break;
        }
    }
    return ans;
}


void DFSInviteLady(vector<vector<int>>& grid, int row, int cnt, int& ans)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    vector<pair<int, int>> vp;

    if (row == m) {
        ans = max(ans, cnt);
        return;
    }
    for (i = 0; i < n; i++) {
        if (grid[row][i] == 1) {
            for (j = row; j < m; j++) {
                if (grid[j][i] == 1) {
                    vp.emplace_back(make_pair(j, i));
                    grid[j][i] = 0;
                }
            }
            DFSInviteLady(grid, row + 1, cnt + 1, ans);
            for (auto p : vp) {
                grid[p.first][p.second] = 1;
            }
        }
    }
    DFSInviteLady(grid, row + 1, cnt, ans);
}
int maximumInvitations1(vector<vector<int>>& grid) // 超时
{
    int i, j;
    int res = 0;
    DFSInviteLady(grid, 0, 0, res);
    return res;
}