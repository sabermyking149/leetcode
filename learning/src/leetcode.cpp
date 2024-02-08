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
#include <cstring>
#include <functional>
#include <numeric>
#include <list>

#include "pub.h"

using namespace std;

/* int gcd(int a, int b)
{
    if(b == 0) {
        return a;
    }
    return gcd(b, a % b);
} */

int TreeNodesNum(TreeNode *root)
{
    if (root == nullptr) {
        return 0;
    }
    return 1 + TreeNodesNum(root->left) + TreeNodesNum(root->right);
}

void initVisited(vector<int>& visited, int val)
{
    for (unsigned int i = 0; i < visited.size(); i++) {
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
    unsigned int i;
    unordered_map<int, unordered_set<int>> e;

    for (i = 0; i < edges.size(); i++) {
        if (edges[i] != -1) {
            e[i].emplace(edges[i]);
        }
    }
    unsigned int n = edges.size();
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
    int i;
    unsigned int j, k;
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


// LC1258
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
void DFSCreateSentences(vector<string>& texts, unsigned int curIdx, unordered_map<string, unordered_set<string>>& synonymsWords,
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
    unsigned int i;
    unordered_set<string> dict;

    for (auto s : synonyms) {
        dict.insert(s[0]);
        dict.insert(s[1]);
    }

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

// 面试题 17.07. 婴儿名字
vector<string> trulyMostPopular(vector<string>& names, vector<string>& synonyms)
{
    unordered_set<string> dict;
    string s1, s2;
    unsigned int i;
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


// LC131
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


// LC2402
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
    unsigned int i;
    priority_queue<int, vector<int>, greater<int>> emptyRooms;
    priority_queue<vector<long long>, vector<vector<long long>>, CMPMostBooked> meetingTimes;
    vector<int> roomUse(n, 0);

    for (i = 0; i < static_cast<unsigned int>(n); i++) {
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
    for (i = 0; i < static_cast<unsigned int>(n); i++) {
        if (roomUse[i] > maxUse) {
            maxUse = roomUse[i];
            ans = i;
        }
    }
    return ans;
}


// LC854
int kSimilarity(string s1, string s2)
{
    unsigned int i, j;
    unsigned int n, m;
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
            unsigned int idx = p.second;
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


// LC505
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
    int i, k;
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
    int i;
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


// LC802
vector<int> eventualSafeNodes(vector<vector<int>>& graph)
{
    int i, j;
    int n = graph.size();
    int m;
    set<int> safeNode;
    unordered_set<int> nodes;
    vector<int> ans;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, unordered_set<int>> reverseConnected;
    for (i = 0; i < n; i++) {
        m = graph[i].size();
        for (j = 0; j < m; j++) {
            edges[i].emplace(graph[i][j]);
            reverseConnected[graph[i][j]].emplace(i);
        }
        nodes.emplace(i);
    }

    while (1) {
        m = nodes.size();
        for (unordered_set<int>::iterator iter = nodes.begin(); iter != nodes.end();) {
            if (edges.count(*iter) == 0) {
                if (reverseConnected.count(*iter) == 0) {
                    safeNode.emplace(*iter);
                    nodes.erase(iter++);
                    continue;
                }
                safeNode.emplace(*iter);
                for (auto it : reverseConnected[*iter]) {
                    if (edges[it].size() > 1) {
                        edges[it].erase(*iter);
                    } else {
                        edges.erase(it);
                    }
                }
                reverseConnected.erase(*iter);
                nodes.erase(iter++);
            } else {
                iter++;
            }
        }
        if (nodes.size() == m) {
            break;
        }
    }
    for (auto it : safeNode) {
        ans.emplace_back(it);
    }
    return ans;
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
    int i;
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

template<>
void Trie<char>::CreateWordTrie(Trie<char> *root, string& word)
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
    unsigned int i;
    int m = words.size();
    Trie<char> *root = new Trie('/');

    for (i = 0; i < m; i++) {
        Trie<char>::CreateWordTrie(root, words[i]);
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


// LC1746
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


void CheckBucket(vector<int>& tasks, int sessionTime, vector<int> &bucket, int idx, bool& isOk)
{
    unsigned int i;
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
bool TrySessions(vector<int>& tasks, int sessionTime, int bucketSize)
{
    bool isOk = false;
    vector<int> bucket(bucketSize, 0);
    CheckBucket(tasks, sessionTime, bucket, 0, isOk);
    return isOk;
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
    unsigned int i;
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
    unsigned int i;

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
    int i;
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


int CheckCanMakeWord(unordered_map<char, int>& alphabet, unordered_map<char, int>& word, vector<int>& score)
{
    int val;
    val = 0;
    for (auto it : word) {
        if (alphabet[it.first] < it.second) {
            return 0;
        }
        val += score[it.first - 'a'] * it.second;
    }
    for (auto it : word) {
        alphabet[it.first] -= it.second;
    }
    return val;
}
void DFSChooseWord(vector<unordered_map<char, int>>& wordsAlphabet, int idx, 
                unordered_map<char, int>& alphabet, vector<int>& score, int curScore, int& ans)
{
    int i;
    int n = wordsAlphabet.size();
    int val;
    if (idx == n) {
        ans = max(curScore, ans);
        return;
    }
    unordered_map<char, int> um;
    um = wordsAlphabet[idx];
    val = CheckCanMakeWord(alphabet, um, score);
    for (i = idx; i < n; i++) {
        DFSChooseWord(wordsAlphabet, i + 1, alphabet, score, curScore + val, ans);
    }
    // 回溯
    if (val != 0) {
        for (auto it : um) {
            alphabet[it.first] += it.second;
        }
    }
}
int maxScoreWords(vector<string>& words, vector<char>& letters, vector<int>& score)
{
    int i;
    int n = words.size();
    int ans;
    bool flag;
    unordered_map<char, int> alphabet;
    for (auto l : letters) {
        alphabet[l]++;
    }
    vector<unordered_map<char, int>> wordsAlphabet;
    unordered_map<char, int> um;
    for (i = 0; i < n; i++) {
        um.clear();
        flag = true;
        for (auto w : words[i]) {
            if (alphabet.find(w) == alphabet.end()) { // 剔除无法拼写单词
                flag = false;
                break;
            }
            um[w]++;
            if (alphabet[w] < um[w]) { // 剔除无法拼写单词
                flag = false;
                break;
            }
        }
        if (flag) {
            wordsAlphabet.emplace_back(um);
        }
    }
    ans = 0;
    n = wordsAlphabet.size();
    for (i = 0; i < n; i++) {
        DFSChooseWord(wordsAlphabet, i, alphabet, score, 0, ans);
    }
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


// dp[i][j] 从i到j的最长偶数子序列
int longestPalindromeSubseq(string s) // 有问题
{
    int i, j;
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (s[i] == s[j]) {
                if (i + 1 < j - 1) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = 2;
                }
            } else {
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[0][n - 1];
}


int maxEqualFreq(vector<int>& nums)
{
    map<int, unordered_set<int>> freqMap;
    unordered_map<int, int> numFreq;
    int cnt, ans;

    cnt = 1;
    for (auto n : nums) {
        numFreq[n]++;
        if (numFreq[n] == 1) {
            freqMap[1].emplace(n);
        } else {
            if (freqMap[numFreq[n] - 1].size() == 1) {
                freqMap.erase(numFreq[n] - 1);
            } else {
                freqMap[numFreq[n] - 1].erase(n);
            }
            freqMap[numFreq[n]].emplace(n);
        }
        if (freqMap.size() == 1 && (freqMap.begin()->second.size() == 1 || freqMap.count(1) == 1)) { // 2 2 2 或 1 2 3
            ans = cnt;
        }
        if (freqMap.size() == 2) {
            auto it2 = freqMap.begin();
            auto it1 = it2++;
            if ((it1->first + 1 == it2->first && it2->second.size() == 1) || 
                (it1->first == 1 && it1->second.size() == 1)) { // 2 2 3 3 4 4 4 或  1 3 3 3 4 4 4
                ans = cnt;
            }
        }
        cnt++;
    }
    return ans;
}


// LC1820 匈牙利算法
bool DFSInviteLady(vector<vector<int>>& grid, unordered_map<int, int>& girlLikeBoy, int boy, int girlsNum, vector<bool>& visited)
{
    int i;
    for (i = 0; i < girlsNum; i++) {
        if (visited[i] == false && grid[boy][i] == 1) {
            visited[i] = true;
            if (girlLikeBoy.count(i) == 0 || DFSInviteLady(grid, girlLikeBoy, girlLikeBoy[i], girlsNum, visited)) {
                girlLikeBoy[i] = boy;
                return true;
            }
        }
    }
    return false;
}
int maximumInvitations(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int res = 0;
    vector<bool> visited(n, false);
    unordered_map<int, int> girlLikeBoy;

    for (i = 0; i < m; i++) {
        visited.assign(n, false);
        if (DFSInviteLady(grid, girlLikeBoy, i, n, visited)) {
            res++;
        }
    }
    return res;
}


vector<int> deckRevealedIncreasing(vector<int>& deck)
{
    int i;
    int t;
    int n = deck.size();
    vector<int> ans;
    deque<int> dq;

    sort (deck.rbegin(), deck.rend());
    dq.push_back(deck[0]);
    for (i = 1; i < n; i++) {
        t = dq.front();
        dq.pop_front();
        dq.push_back(t);
        dq.push_back(deck[i]);
    }
    while (!dq.empty()) {
        ans.emplace_back(dq.back());
        dq.pop_back();
    }
    return ans;
}


int longestWPI(vector<int>& hours)
{
    int i, j;
    int n = hours.size();
    int ans;
    vector<int> v;
    vector<vector<int>> vv;
    for (auto h : hours) {
        if (h > 8) {
            v.emplace_back(1);
        } else {
            v.emplace_back(-1);
        }
    }
    vector<int> prefixSum(n, 0);
    prefixSum[0] = v[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + v[i];
    }

    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(prefixSum[i], i));
    }
    sort (vp.begin(), vp.end());
    ans = 0;
    v.clear();
    v.emplace_back(vp[0].second);
    for (i = 1; i < n; i++) {
        if (vp[i].first == vp[i - 1].first) {
            v.emplace_back(vp[i].second);
        } else {
            vv.emplace_back(v);
            v.clear();
            v.emplace_back(vp[i].second);
        }
    }
    vv.emplace_back(v);
    // cout << vv.size() << endl;
    n = vv.size();
    for (i = 0; i < n - 1; i++) {
        if (prefixSum[vv[i][vv[i].size() - 1]] > 0) {
            ans = max(ans, vv[i][vv[i].size() - 1] + 1);
        }
        for (j = i + 1; j < n; j++) {
            if (prefixSum[vv[j][vv[j].size() - 1]] > 0) {
                ans = max(ans, vv[j][vv[j].size() - 1] + 1);
            } else {
                ans = max(vv[j][vv[j].size() - 1] - vv[i][0], ans);
            }
        }
    }
    return ans;
}


int maxWidthRamp(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int ans, minIdx;
    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(nums[i], i));
    }
    sort (vp.begin(), vp.end());
    ans = 0;
    minIdx = vp[0].second;
    for (i = 1; i < n; i++) {
        if (vp[i].second > minIdx) {
            ans = max(ans, vp[i].second - minIdx);
        }
        minIdx = min(minIdx, vp[i].second);
    }
    return ans;
}


// LC1186
// dp[i][j]  以i结尾是否已删除一个数字的最大子数组和 j = 0 未删 j = 1 已删
int maximumSum(vector<int>& arr)
{
    int i;
    int n = arr.size();
    int a, b, ans;
    vector<vector<int>> dp(n, vector<int>(2, 0));

    dp[0][0] = arr[0];
    ans = arr[0];
    for (i = 1; i < n; i++) {
        dp[i][0] = (dp[i - 1][0] < 0 ? arr[i] : dp[i - 1][0] + arr[i]);
        a = dp[i - 1][0]; // arr[i] 删除
        b = (dp[i - 1][1] < 0 ? arr[i] : dp[i - 1][1] + arr[i]); // arr[i] 不删除
        dp[i][1] = max(a, b);
        ans = max(ans, max(dp[i][0], dp[i][1]));
    }
    return ans;
}


// LC1011
int shipWithinDays(vector<int>& weights, int days)
{
    int i;
    int n = weights.size();
    int sum = 0;
    int cnt;
    int left, right, mid;

    left = weights[0];
    for (auto w : weights) {
        sum += w;
        left = max(left, w);
    }
    right = sum;
    while (left <= right) { // 所求为left
        mid = (right - left) / 2 + left;
        cnt = 0;
        sum = 0;
        for (i = 0; i < n ; i++) {
            sum += weights[i];
            if (sum > mid) {
                sum = weights[i];
                cnt++;
            }
        }
        cnt++;
        if (cnt > days) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


int func(long long n)
{
    int t;
    int mod = 1000000007;
    long long ans = 0;

    while (n > 1) {
        t = ceil(log10(n) / log10(2));
        ans = (ans + (n - static_cast<long long>(pow(2, t - 1))) * t) % mod;
        n = pow(2, t - 1);
    }
    return ans;
}


// LC2571
int minOperations(int n)
{
    int i, j;
    int step;
    int k, t;
    vector<int> nums = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    unordered_set<int> visited;
    queue<int> q;

    step = 0;
    q.push(n);
    visited.emplace(n);
    while (!q.empty()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            t = q.front();
            // cout << t << " ";
            // visited.emplace(t);
            q.pop();
            for (j = 0; j < nums.size(); j++) {
                if (t == nums[j]) {
                    return step + 1;
                }
                if (t - nums[j] > 0 && visited.count(t - nums[j]) == 0) {
                    q.push(t - nums[j]);
                    visited.emplace(t - nums[j]);
                }
                if (t + nums[j] < 131072 && visited.count(t + nums[j]) == 0) {
                    q.push(t + nums[j]);
                    visited.emplace(t + nums[j]);
                }
            }
        }
        step++;
    }
    return -1;
}


// LC898
int subarrayBitwiseORs(vector<int>& arr)
{
    int i, j;
    int n = arr.size();
    unordered_set<int> s;

    for (i = 0; i < n; i++) {
        s.emplace(arr[i]);
        for (j = i - 1; j >= 0; j--) {
            if ((arr[i] | arr[j]) == arr[j]) {
                break;
            }
            arr[j] |= arr[i];
            s.emplace(arr[j]);
        }
    }
    return s.size();
}


// LC444
bool sequenceReconstruction(vector<int>& nums, vector<vector<int>>& sequences)
{
    int i;
    unordered_map<int, unordered_set<int>> edges;
    for (auto seq : sequences) {
        for (i = 1; i < seq.size(); i++) {
            edges[seq[i - 1]].emplace(seq[i]);
        }
    }
    for (i = 1; i < nums.size(); i++) {
        if (edges[nums[i - 1]].count(nums[i]) == 0) {
            return false;
        }
    }
    return true;
}


// LC2552
// 0 <= i < j < k < l < n
// nums[i] < nums[k] < nums[j] < nums[l]
/*long long countQuadruplets(vector<int>& nums)
{
    long long ans;
    // long long left, right;
    unsigned int n = nums.size();
    vector<vector<int>> left(n, vector<int>(n, 0)), right(n, vector<int>(n, 0));
    // vector<int> left(n, 0), right(n, 0);
    int j, k;

    j = 1;
    k = n - 2;
    // left = right = 0;
    if (nums[0] < nums[k]) {
        left[1][k] = 1;
    }
    if (nums[n - 1] > nums[j]) {
        right[n - 1][j] = 1;
    }
    for ()
    ans = 0;
    for (j = 1; j < n; j++) {
        for (k = n - 2; k >= 0; k--) {
            if (j >= k) {
                break;
            }
            if (nums[k] >= nums[j]) {
                continue;
            }

            ans += static_cast<long long>(left[j][k]) * right[k][j];
        }
    }
    return ans;
}*/


// LC1131
int maxAbsValExpr(vector<int>& arr1, vector<int>& arr2)
{
    int i;
    int n = arr1.size();
    int minVal, ans;

    ans = 0;
    // arr1[i] + arr2[i] + i
    minVal = arr1[0] + arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, arr1[i] + arr2[i] + i - minVal);
        minVal = min(minVal, arr1[i] + arr2[i] + i);
    }
    // -arr1[i] + arr2[i] + i;
    minVal = -arr1[0] + arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, -arr1[i] + arr2[i] + i - minVal);
        minVal = min(minVal, -arr1[i] + arr2[i] + i);
    }
    // arr1[i] - arr2[i] + i;
    minVal = arr1[0] - arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, arr1[i] - arr2[i] + i - minVal);
        minVal = min(minVal, arr1[i] - arr2[i] + i);
    }
    // -arr1[i] - arr2[i] + i;
    minVal = -arr1[0] - arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, -arr1[i] - arr2[i] + i - minVal);
        minVal = min(minVal, -arr1[i] - arr2[i] + i);
    }
    return ans;
}


// LC310
int curLongestRouteLen = 0;
vector<int> longestRoute;
void DFSSearchLongestRourte(unordered_map<int, unordered_set<int>>& edge, int cur, vector<bool>& visited, vector<int>& route, bool sign, bool &quit)
{
    if (quit) {
        return;
    }
    visited[cur] = true;
    route.push_back(cur);
    if (sign == false) {
        if (curLongestRouteLen < route.size()) {
            curLongestRouteLen = route.size();
        }
    } else {
        if (route.size() == curLongestRouteLen) {
            longestRoute = route;
            quit = true;
            return;
        }
    }
    for (auto it : edge[cur]) {
        if (visited[it] == false) {
            DFSSearchLongestRourte(edge, it, visited, route, sign, quit);
        }
    }
    route.pop_back();
}
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges)
{
    int i;
    int node, num;
    int lastNode;
    vector<int> ans;
    vector<bool> visited(n, false);
    unordered_map<int, unordered_set<int>> edge;
    if (n == 1) {
        return {0};
    }
    for (auto e : edges) {
        edge[e[0]].emplace(e[1]);
        edge[e[1]].emplace(e[0]);
    }

    int dist;
    queue<int> q;

    dist = 0;
    q.push(0);
    while (q.size()) {
        num = q.size();
        for (i = 0; i < n; i++) {
            node = q.front();
            q.pop();
            visited[node] = true;
            for (auto it : edge[node]) {
                if (visited[it] == false) {
                    q.push(it);
                    lastNode = it;
                }
            }
        }
        dist++;
    }
    // cout << lastNode;
    // 从lastNode找最长路径
    visited.assign(n, false);
    vector<int> route;
    bool sign = false;
    bool quit = false;
    DFSSearchLongestRourte(edge, lastNode, visited, route, sign, quit);
    sign = true;
    visited.assign(n, false);
    DFSSearchLongestRourte(edge, lastNode, visited, route, sign, quit);
    num = longestRoute.size();
    if (num % 2 == 1) {
        ans = {longestRoute[num / 2]};
    } else {
        ans = {longestRoute[num / 2 - 1], longestRoute[num / 2]};
    }
    return ans;
}


// LC210
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites)
{
    int i;
    int n, t;
    vector<int> ans;
    vector<int> degree(numCourses, 0);
    unordered_map<int, unordered_set<int>> edge;
    for (auto p : prerequisites) {
        degree[p[0]]++;
        edge[p[1]].emplace(p[0]);
    }
    queue<int> q;
    for (i = 0; i < numCourses; i++) {
        if (degree[i] == 0) {
            q.push(i);
        }
    }
    if (q.empty()) {
        return {};
    }
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            ans.emplace_back(t);
            if (edge.count(t) == 0) {
                continue;
            }
            for (auto it : edge[t]) {
                degree[it]--;
                if (degree[it] == 0) {
                    q.push(it);
                }
            }
        }
    }
    if (ans.size() != numCourses) {
        return {};
    }
    return ans;
}


// LC2576
int maxNumOfMarkedIndices(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int ans, t;
    int left, right, mid;
    vector<bool> visited(n, false);

    sort(nums.begin(), nums.end());
    ans = 0;
    left = n / 2;
    for (i = 0; i < n / 2; i++) {
        right = n - 1;
        t = nums[i] * 2;
        while (left <= right) { // 所求为left
            mid = (right - left) / 2 + left;
            if (nums[mid] >= t) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (left >= n) {
            break;
        }
        if (visited[left] != false) {
            for (j = left; j < n; j++) {
                if (visited[j] == false) {
                    break;
                }
            }
            if (j == n + 1) {
                break;
            } else {
                left = j;
            }
        }
        visited[left] = true;
        // cout << nums[i] << " " << nums[left] << " " << left << endl;
        ans += 2;

        left++;
        if (left >= n) {
            break;
        }
    }
    return ans;
}


// 面试题 05.02
string printBin(double num)
{
    string ans = "0.";
    double carry = 0.5;
    while (carry > 0.0078125) {
        if (num >= carry) {
            ans += '1';
            num -= carry;
            if (fabs(num) < 10e-7) {
                return ans;
            }
        } else {
            ans += '0';
        }
        carry /= 2;
    }
    return "ERROR";
}


// 面试题 08.02
void DFSFindPath(vector<vector<int>>& obstacleGrid, int row, int col, vector<vector<int>>& route, vector<vector<int>>& ans, bool& find)
{
    if (find) {
        return;
    }
    int i;
    int m, n;
    int nrow, ncol;
    int directions[2][2] = {{0, 1}, {1, 0}};

    m = obstacleGrid.size();
    n = obstacleGrid[0].size();
    route.push_back({row, col});
    if (row == m - 1 && col == n - 1) {
        find = true;
        ans = route;
        return;
    }
    for (i = 0; i < 2; i++) {
        nrow = row + directions[i][0];
        ncol = col + directions[i][1];

        if (nrow >= 0 && nrow < m && ncol >= 0 && ncol < n && obstacleGrid[nrow][ncol] == 0) {
            obstacleGrid[nrow][ncol] = 1;
            DFSFindPath(obstacleGrid, nrow, ncol, route, ans, find);
            // obstacleGrid[nrow][ncol] = 0; 此处回溯会超时
        }
    }
    route.pop_back();
}
vector<vector<int>> pathWithObstacles(vector<vector<int>>& obstacleGrid)
{
    vector<vector<int>> ans;
    vector<vector<int>> route;
    vector<vector<int>> grid;
    bool find = false;
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();

    if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) {
        return ans;
    }
    grid = obstacleGrid;
    DFSFindPath(grid, 0, 0, route, ans, find);
    return ans;
}


// LC982
int countTriplets(vector<int>& nums)
{
    int i, j;
    int ans;
    int n;
    unordered_map<int, int> andCnt;

    n = nums.size();
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            andCnt[(nums[i] & nums[j])]++;
        }
    }
    ans = 0;
    for (auto it : andCnt) {
        for (auto n : nums) {
            if ((n & it.first) == 0) {
                ans += it.second;
            }
        }
    }
    return ans;
}


class LC2581 {
public:
    int ans = 0;
    map<long long, int> condition;
    map<long long, int> curUsedCondition;
    void DFS(unordered_map<int, unordered_set<int>>& gridedges, size_t cur, size_t parent, int& curk)
    {
        // cout << curk << endl;
        long long t = parent << 32 | cur;
        if (condition.count(t) == 1) {
            // printf ("[%d %d]\n", parent, cur);
            curk += condition[t];
            curUsedCondition[t] = condition[t];
        }
        for (auto e : gridedges[cur]) {
            if (e != parent) {
                DFS(gridedges, e, cur, curk);
            }
        }
    }
    void ChangeRoot(unordered_map<int, unordered_set<int>>& gridedges, size_t cur, size_t parent, int curk, int k)
    {
        long long t = cur << 32 | parent;
        if (condition.count(t) == 1 && curUsedCondition.count(t) == 0) {
            curk += condition[t];
            curUsedCondition[t] += condition[t];
        }
        t = parent << 32 | cur;
        if (condition.count(t) == 1 && curUsedCondition.count(t) == 1) {
            curk -= condition[t];
            curUsedCondition.erase(t);
        }
        if (curk >= k) {
            ans++;
        }
        for (auto it : gridedges[cur]) {
            if (it != parent) {
                ChangeRoot(gridedges, it, cur, curk, k);
            }
        }
    }
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k)
    {
        for (auto g : guesses) {
            condition[(long long)g[0] << 32 | g[1]]++;
        }
        unordered_map<int, unordered_set<int>> gridedges;
        for (auto e : edges) {
            gridedges[e[0]].emplace(e[1]);
            gridedges[e[1]].emplace(e[0]);
        }
        int i;
        int n = edges.size() + 1;
        int curk;

        curk = 0;
        DFS(gridedges, 0, -1, curk);
        if (curk >= k) {
            ans++;
        }
        for (auto it : gridedges[0]) {
            ChangeRoot(gridedges, it, 0, curk, k);
        }
        return ans;
    }
};


// LC1599
int minOperationsMaxProfit(vector<int>& customers, int boardingCost, int runningCost)
{
    int ans;
    int curProfit, t;
    int curTime, curCustomers;

    curTime = 0;
    curCustomers = 0;
    ans = -1;
    t = INT_MIN;
    curProfit = 0;
    while (1) {
        if (curTime < customers.size()) {
            curCustomers += customers[curTime];
        }
        if (curCustomers >= 4) {
            curProfit += 4 * boardingCost - runningCost;
            curCustomers -= 4;
        } else {
            curProfit += curCustomers * boardingCost - runningCost;
            curCustomers = 0;
        }
        if (curProfit > t) {
            t = curProfit;
            ans = curTime + 1;
        }
        curTime++;
        if (curTime >= customers.size() && curCustomers == 0) {
            break;
        }
    }
    return curProfit <= 0 ? -1 : ans;
}


// 面试题 16.09. 运算
class Operations {
public:
    Operations() {

    }
    long negative(long n)
    {
        if (n == 0) {
            return 0;
        }
        bool f = false;
        long base = -1;
        if (n < 0) {
            f = true;
            base = 1;
        }
        long i = 0;
        while (1) {
            if (f) { 
                if (i + base + n > 0) {
                    base = 1;
                } else if (i + base + n == 0) {
                    i += base;
                    break;
                } else {
                    i += base;
                    base += base;
                }
            } else {
                if (i + base + n < 0) {
                    base = -1;
                } else if (i + base + n == 0) {
                    i += base;
                    break;
                } else {
                    i += base;
                    base += base;
                }
            }
        }
        return i;
    }
    int minus(int a, int b)
    {
        return a + negative(b);
    }
    
    int multiply(int a, int b)
    {
        if (a == 0 || b == 0) {
            return 0;
        }
        bool f = false;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
            f  = true;
        }
        long aa, bb;
        if (a < 0) {
            aa = negative(a);
        } else {
            aa = a;
        }
        if (b < 0) {
            bb = negative(b);
        } else {
            bb = b;
        }

        long ans = 0;
        long base = 1;
        long i = 0;
        long t = aa;
        while (1) {
            if (i + base > bb) {
                base = 1;
                t = aa;
            } else if (i + base == bb) {
                i += base;
                ans += t;
                break;
            } else {
                ans += t;
                i += base;
                base += base;
                t += t;
            }
        }
        if (f) {
            return negative(ans);
        }
        return ans;
    }
    long my_multiply(long a, long b)
    {
        if (a == 0 || b == 0) {
            return 0;
        }
        long ans = 0;
        long base = 1;
        long i = 0;
        long t = a;
        while (1) {
            if (i + base > b) {
                base = 1;
                t = a;
            } else if (i + base == b) {
                i += base;
                ans += t;
                break;
            } else {
                ans += t;
                i += base;
                base += base;
                t += t;
            }
        }
        return ans;
    }
    long GetHalfOfNum(long n)
    {
        long i;
        long base;

        base = 1;
        i = 0;
        while (1) {
            if (i + base + i + base > n) {
                base = 1;
            } else if (i + base + i + base == n || i + base + i + base + 1 == n) {
                i += base;
                break;
            } else {
                i += base;
                base += base;
            }
        }
        return i;
    }
    int divide(int a, int b)
    {
        if (a == 0) {
            return 0;
        } else if (b == 1) {
            return a;
        } else if (b == -1) {
            return negative(a);
        }
        bool f = false;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
            f  = true;
        }
        long aa, bb;
        if (a < 0) {
            aa = negative(a);
        } else {
            aa = a;
        }
        if (b < 0) {
            bb = negative(b);
        } else {
            bb = b;
        }
        if (bb > aa) {
            return 0;
        }
        long left, right, mid;
        long t;

        left = 1;
        right = aa;
        while (left <= right) {  // 所求为right
            mid = GetHalfOfNum(left + right);
            t = my_multiply(mid, bb);
            if (t > aa) {
                right = minus(mid, 1);
            } else if (t < aa) {
                left = mid + 1;
            } else {
                if (f) {
                    return negative(mid);
                }
                return mid;
            }
        }
        if (f) {
            return negative(right);
        }
        return right;
    }
};


// 面试题 17.05. 字母与数字
vector<string> findLongestSubarray(vector<string>& array)
{
    int i;
    int n = array.size();
    int t;
    int curLongest;
    map<int, pair<int, int>> data;
    map<int, vector<int>> diffData;
    pair<int, int> ansIdx = {-1, -1};
    vector<string> ans;

    curLongest = 0;
    data[0] = {0, 0};
    for (i = 0; i < n; i++) {
        if (i != 0) {
            data[i] = data[i - 1];
        }
        if (isalpha(array[i][0])) {
            data[i].first++;
            diffData[data[i].first - data[i].second].emplace_back(i);
        } else {
            data[i].second++;
            diffData[data[i].first - data[i].second].emplace_back(i);
        }
        if (data[i].first == data[i].second) {
            if (i + 1 > curLongest) {
                curLongest = i + 1;
                ansIdx = {0, i};
            }
        } else {
            t = data[i].first - data[i].second;
            if (diffData.count(t) == 1 && diffData[t].size() != 1) {
                if (i - diffData[t][0] > curLongest) {
                    curLongest = i - diffData[t][0];
                    ansIdx = {diffData[t][0] + 1, i};
                }
            }
        }
    }
    if (ansIdx.first == -1) {
        return {};
    }
    for (i = ansIdx.first; i <= ansIdx.second; i++) {
        ans.emplace_back(array[i]);
    }
    return ans;
}


// LC1617
void DFSCalcSubTree(unordered_map<int, unordered_set<int>>& treeedges, int cur, int parent, 
    set<int>& nodes, int dist, int& maxDist, set<int>& visited)
{
    visited.emplace(cur);
    maxDist = max(dist, maxDist);
    for (auto it : treeedges[cur]) {
        if (it != parent && nodes.count(it) == 1) {
            DFSCalcSubTree(treeedges, it, cur, nodes, dist + 1, maxDist, visited);
        }
    }
}
void ChooseNode(int n, int idx, int cnt,int tolcnt, vector<int>& chosenNode, vector<set<int>>& nodes)
{
    set<int> node;
    if (tolcnt == cnt) {
        for (auto n : chosenNode) {
            // cout << n << " ";
            node.emplace(n);
        }
        // cout << endl;
        nodes.emplace_back(node);
        return;
    }
    int i;
    for (i = idx; i < n; i++) {
        chosenNode.emplace_back(i + 1);
        ChooseNode(n, i + 1, cnt + 1, tolcnt, chosenNode, nodes);
        chosenNode.pop_back();
    }
}
vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges)
{
    int i, j, k;
    int maxDist;
    vector<int> ans(n - 1, 0);
    unordered_map<int, unordered_set<int>> treeedges;
    vector<int> chosenNode; 
    unordered_map<int, int> subTreeDistCount;
    vector<set<int>> nodes;
    set<int> visited;
    // bool f = false; // 是否所有点都能有效访问
    for (auto e : edges) {
        treeedges[e[0]].emplace(e[1]);
        treeedges[e[1]].emplace(e[0]);
    }
    for (i = 2; i <= n; i++) {
        nodes.clear();
        ChooseNode(n, 0, 0, i, chosenNode, nodes); // 枚举所有点的集合
        for (j = 0; j < nodes.size(); j++) {
            /* 超时
            maxDist = 0;
            for (auto it : nodes[j]) {
                visited.clear();
                f = false;
                DFSCalcSubTree(treeedges, it, -1, nodes[j], 0, maxDist, visited);
                if (visited != nodes[j]) {
                    break;
                }
                f = true;
            }
            if (f) {
                subTreeDistCount[maxDist]++;
            }
            */
            // 两次BFS求最大距离
            int t;
            int m;
            int lastNode;
            queue<int> q;
            visited.clear(); 
            q.push(*nodes[j].begin());
            while (q.size()) {
                m = q.size();
                for (k = 0; k < m; k++) {
                    t = q.front();
                    lastNode = t;
                    q.pop();
                    visited.emplace(t);
                    for (auto it : treeedges[t]) {
                        if (nodes[j].count(it) == 1 && visited.count(it) == 0) {
                            q.push(it);
                            //lastNode = it;
                        }
                    }
                }
            }
            // 判断点集是否能组成子树
            if (visited != nodes[j]) {
                continue;
            }
            //cout << lastNode << endl;
            int dist;

            q.push(lastNode);
            visited.clear();
            dist = 0;
            while (q.size()) {
                m = q.size();
                for (k = 0; k < m; k++) {
                    t = q.front();
                    q.pop();
                    visited.emplace(t);
                    for (auto it : treeedges[t]) {
                        if (nodes[j].count(it) == 1 && visited.count(it) == 0) {
                            q.push(it);
                            // lastNode = it;
                        }
                    }
                }
                if (q.size() != 0)
                    dist++;
            }

            subTreeDistCount[dist]++;
            // for (auto it : nodes[j]) {cout << it << " ";} cout << dist << endl;
        }
    }
    for (auto it : subTreeDistCount ) {
    ans[it.first - 1] = it.second;
    }
    return ans;
}


// 面试题 16.13. 平分正方形
vector<vector<double>> CalcDot(double x, vector<int>& square, pair<double, double>& center, double k)
{
    double y;
    double x1, y1;

    y = k * (x - center.first) + center.second;
    x1 = x + square[2];
    y1 = k * (x1 - center.first) + center.second;
    if (k > 0 && y < square[1]) {
        y = square[1];
        x = (y - center.second) / k + center.first;
        y1 = square[1] + square[2];
        x1 = (y1 - center.second) / k + center.first;
    }
    if (k < 0 && y > square[1] + square[2]) {
        y = square[1] + square[2];
        x = (y - center.second) / k + center.first;
        y1 = square[1];
        x1 = (y1 - center.second) / k + center.first;
    }
    return {{x, y}, {x1, y1}};
}
vector<double> cutSquares(vector<int>& square1, vector<int>& square2)
{
    double k;
    pair<double, double> dot1, dot2;

    dot1 = {square1[0] + square1[2] * 1.0 / 2, square1[1] + square1[2] * 1.0 / 2};
    dot2 = {square2[0] + square2[2] * 1.0 / 2, square2[1] + square2[2] * 1.0 / 2};

    vector<double> ans(4, 0);
    if (fabs(dot1.first - dot2.first) < 10e-5) {
        ans[0] = dot1.first;
        ans[1] = max(dot1.second + square1[2] * 1.0 / 2,
            dot2.second + square2[2] * 1.0 / 2);
        ans[2] = dot2.first;
        ans[3] = min(dot1.second - square1[2] * 1.0 / 2,
            dot2.second - square2[2] * 1.0 / 2);
        if (ans[1] > ans[3]) {
            swap(ans[1], ans[3]);
        }
        return ans;
    }
    k = (dot1.second - dot2.second) / (dot1.first - dot2.first);

    // 分别求直线穿过的4个点
    vector<vector<double>> t1, t2;
    t1 = CalcDot(square1[0], square1, dot1, k);
    t2 = CalcDot(square2[0], square2, dot2, k);
    t1.insert(t1.end(), t2.begin(), t2.end());
    sort(t1.begin(), t1.end());
    return {t1[0][0], t1[0][1], t1[3][0], t1[3][1]};
}


// LC2597
void ChooseNode(int n, int idx, int cnt, int tolcnt, int numsArr[], int numsArrSize, int chosenNode[], int k, int& ans)
{
    if (tolcnt - cnt > numsArrSize - idx) {
        return;
    }
    if (tolcnt == cnt) {
        ans++;
        return;
    }
    int i, j;
    for (i = idx; i < numsArrSize; i++) {
        if (cnt > 0) {
            for (j = 0; j < cnt; j++) {
                if (abs(chosenNode[j] - numsArr[i]) == k) {
                    break;
                }
            }
            if (j != cnt) {
                continue;
            }
        }
        chosenNode[cnt] = numsArr[i];
        ChooseNode(n, i + 1, cnt + 1, tolcnt, numsArr, numsArrSize, chosenNode, k, ans);
    }
}
int beautifulSubsets(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int chosenNode[20] = {0};
    int numsArr[20] = {0};
    for (i = 0; i < n; i++) {
        numsArr[i] = nums[i];
    }
    int ans = 0;
    for (i = 2; i <= n; i++) {
        ChooseNode(n, 0, 0, i, numsArr, n, chosenNode, k, ans);
    }
    return ans + n;
}
int beautifulSubsets_1(vector<int>& nums, int k) // 动态规划
{
    int i;
    int n = nums.size();
    int mod, t;
    int ans;
    vector<int> f(n + 1, 0);
    vector<int> v;
    unordered_map<int, vector<int>> modK;
    unordered_map<int, int> numCnt;

    sort(nums.begin(), nums.end()); 
    for (i = 0; i < n; i++) {
        numCnt[nums[i]]++;
        mod = nums[i] % k;
        if (modK.count(mod) == 0) {
            modK[mod].emplace_back(nums[i]);
        } else {
            t = modK[mod][modK[mod].size() - 1];
            if (t != nums[i]) {
                modK[mod].emplace_back(nums[i]);
            }
        }
    }
    ans = 0;
    vector<int> sum;
    for (auto it : modK) {
        v = it.second;
        f[0] = 1;
        f[1] = pow(2, numCnt[it.second[0]]);
        for (i = 1; i < v.size(); i++) {
            if (v[i] - v[i - 1] != k) {
                f[i + 1] = f[i] * pow(2, numCnt[v[i]]);
            } else {
                f[i + 1] = f[i] + f[i - 1] * (pow(2, numCnt[v[i]]) - 1);
            }
        }
        // cout << f[v.size()] << endl;;
        if (sum.size() == 0) {
            sum.emplace_back(f[v.size()] - 1);
        } else {
            n = sum.size();
            for (i = 0; i < n; i++) {
                sum.emplace_back(sum[i] * (f[v.size()] - 1));
            }
            sum.emplace_back(f[v.size()] - 1);
        }
    }
    for (auto s : sum) {
        ans += s;
    }
    return ans;
}
int beautifulSubsets_2(vector<int>& nums, int k) // 动态规划,更好方法
{
    int i;
    int n = nums.size();
    int mod, t;
    int ans;
    vector<int> f(n + 1, 0);
    vector<int> v;
    unordered_map<int, vector<int>> modK;
    unordered_map<int, int> numCnt;

    sort(nums.begin(), nums.end()); 
    for (i = 0; i < n; i++) {
        numCnt[nums[i]]++;
        mod = nums[i] % k;
        if (modK.count(mod) == 0) {
            modK[mod].emplace_back(nums[i]);
        } else {
            t = modK[mod][modK[mod].size() - 1];
            if (t != nums[i]) {
                modK[mod].emplace_back(nums[i]);
            }
        }
    }
    ans = 1;
    for (auto it : modK) {
        v = it.second;
        f[0] = 1;
        f[1] = pow(2, numCnt[it.second[0]]);
        for (i = 1; i < v.size(); i++) {
            if (v[i] - v[i - 1] != k) {
                f[i + 1] = f[i] * pow(2, numCnt[v[i]]);
            } else {
                f[i + 1] = f[i] + f[i - 1] * (pow(2, numCnt[v[i]]) - 1);
            }
        }
        // cout << f[v.size()] << endl;
        ans *= f[v.size()];
    }
    return ans - 1;
}

// LC1012
int numDupDigitsAtMostN(int n)
{
    int i, k;
    int t, x, cnt, countUnique;
    string str;
    vector<int> f(10, 0), prefixSum(n, 0);

    f[1] = 9;
    k = 9;
    for (i = 2; i <= 9; i++) {
        f[i] = f[i - 1] * k;
        k--;
    }
    prefixSum[1] = f[1];
    for (i = 2; i <= 9; i++) {
        prefixSum[i] = prefixSum[i - 1] + f[i];
    }
    t = n;
    countUnique = 0;
    str = to_string(n);
    
    k = str.size();
    //if (str[0] > '1') {
        countUnique += prefixSum[k - 1];
    //}
    t = k;
    for (i = 0; i < k; i++) {
        if (str[i] == '0') {
            continue;
        }
        x = str[i] - '0' - 1;
        if (i == k - 1) {
            x++;
        }
        countUnique += (f[t] / 9 * x);
        t--;
    }
    return n - countUnique;
}


// LC127
bool IsConnectedWord(string& a, string& b)
{
    int i;
    int n = a.size();
    int cnt;

    cnt = 0;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            cnt++;
        }
    }
    return cnt == 1;
}
int ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
    int i, j;
    int n = wordList.size();
    int ans;
    string t;
    unordered_set<string> visited;
    unordered_map<string, unordered_set<string>> edges;

    for (i = 0; i < n; i++) {
        if (wordList[i] == beginWord) {
            break;
        }
    }
    if (i == n) {
        wordList.emplace_back(beginWord);
        n = wordList.size();
    }
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (IsConnectedWord(wordList[i], wordList[j])) {
                edges[wordList[i]].emplace(wordList[j]);
                edges[wordList[j]].emplace(wordList[i]);
            }
        }
    }
    /* for (auto it1 : edges) {
        cout << it1.first << ":";
        for (auto it2 : it1.second) {
            cout << it2 << " ";
        }
        cout << endl;
    } */
    if (edges.count(endWord) == 0) {
        return 0;
    }

    queue<string> q;
    q.push(beginWord);
    ans = 1;
    while (!q.empty()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            if (t == endWord) {
                return ans;
            }
            visited.emplace(t);
            for (auto it : edges[t]) {
                if (visited.count(it) == 0) {
                    q.push(it);
                }
            }
        }
        ans++;
    }
    return 0;
}


// LC820
int minimumLengthEncoding(vector<string>& words)
{
    int n = words.size();
    int m;
    int i, j, k;
    vector<bool> canShorten(n, false);
    for (i = 0; i < n; i++) {
        if (canShorten[i]) {
            continue;
        }
        for (j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }
            if (words[i].size() > words[j].size()) {
                continue;
            }
            if (canShorten[j]) {
                continue;
            }
            m = words[j].size();
            for (k = words[i].size() - 1; k >= 0; k--) {
                if (words[i][k] != words[j][m - 1]) {
                    break;
                }
                m--;
            }
            if (k == -1) {
                canShorten[i] = true;
                break;
            }
        }
    }
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (canShorten[i] == false) {
            ans += words[i].size() + 1;
        }
    }
    return ans;
}
void DFSScanTrieTree(Trie<char> *node, int curLen, int& ans)
{
    int i;
    if (node->children.size() == 0) {
        ans += curLen + 1;
        return;
    }
    for (i = 0; i < node->children.size(); i++) {
        DFSScanTrieTree(node->children[i], curLen + 1, ans);
    }
}
int minimumLengthEncoding_1(vector<string>& words) // 后缀字典树
{
    int ans = 0;
    Trie<char> *root = new Trie<char>('/');
    for (auto word : words) {
        reverse(word.begin(), word.end());
        Trie<char>::CreateWordTrie(root, word);
    }
    DFSScanTrieTree(root, 0, ans);
    return ans;
}


// LC2305
void DFSGetFullPermutation(int n, vector<int>& record, vector<bool>& visited, vector<vector<int>>& ans)
{
    int i;
    if (record.size() == n) {
        ans.emplace_back(record);
        return;
    }
    for (i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        visited[i] = true;
        record.emplace_back(i);
        DFSGetFullPermutation(n, record, visited, ans);
        visited[i] = false;
        record.pop_back();
    }
}
void DFSTrySplitWays(string& t, int idx, int cnt, vector<string>& ans)
{
    int i;
    int n = t.size();
    if (n - idx < cnt) {
        return;
    }
    if (cnt == 0) {
        ans.emplace_back('0' + t);
        return;
    }
    for (i = idx; i < n; i++) {
        t[i] = '1';
        DFSTrySplitWays(t, i + 1, cnt - 1, ans);
        t[i] = '0';
    }
}
int distributeCookies(vector<int>& cookies, int k)
{
    int i, j, l;
    int ans, curSum, unfairness;
    int n = cookies.size();
    string t(n - 1, '0');
    vector<bool> visited(n, false);

    vector<string> SplitWays;
    vector<int> record;
    vector<vector<int>> FullPermutation;

    DFSTrySplitWays(t, 0, k - 1, SplitWays);
    DFSGetFullPermutation(n, record, visited, FullPermutation);

    int pSize = FullPermutation.size();
    int swSize = SplitWays.size();

    ans = INT_MAX;
    for (i = 0; i < pSize; i++) {
        for (j = 0; j < swSize; j++) {
            curSum = 0;
            unfairness = 0;
            for (l = 0; l < n; l++) {
                if (SplitWays[j][l] == '0') {
                    curSum += cookies[FullPermutation[i][l]];
                } else {
                    unfairness = max(unfairness, curSum);
                    curSum = cookies[FullPermutation[i][l]];
                }
            }
            unfairness = max(unfairness, curSum);
            ans = min(ans, unfairness);
        }
    }
    return ans;
}


// LC711
void ScanGraphAndRecord(vector<vector<int>>& grid, int row, int col, vector<vector<bool>> &visited, vector<pair<int, int>>& points)
{
    if (row < 0 || row == grid.size() || col < 0 || col == grid[0].size() || grid[row][col] == 0 || visited[row][col] == true)
        return;

    int i;
    int direction[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

    visited[row][col] = true;
    points.push_back({row, col});
    for (i = 0; i < 4; i++)
        ScanGraphAndRecord(grid, row + direction[i][0], col + direction[i][1], visited, points);
}
int numDistinctIslands2(vector<vector<int>>& grid)
{
    int i, j, k, l;
    int minR, minC;
    set<string> shape;
    
    if (grid.size() == 0)
        return 0;

    string shapeStr;
    vector<vector<bool>> visited(grid.size(),vector<bool>(grid[0].size(), false));
    vector<pair<int, int>> points, t;
    vector<string> shapeStrColl;
    vector<vector<int>> pontsTrans = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (i = 0; i < grid.size(); i++) {
        for (j = 0; j < grid[0].size(); j++) {
            if (grid[i][j] == 1 && visited[i][j] == false) {
                points.clear();
                shapeStrColl.clear();
                ScanGraphAndRecord(grid, i, j, visited, points);

                // 求points的8种变换坐标
                // 即 (x,y),(−x,y),(x,−y),(−x,−y),(y,x),(−y,x),(y,−x),(−y,−x)
                t = points;
                for (l = 0; l < 4; l++) {
                    for (k = 0; k < points.size(); k++) {
                        t[k].first = points[k].first * pontsTrans[l][0];
                        t[k].second = points[k].second * pontsTrans[l][1];
                    }
                    // 变换后的相对坐标
                    minR = INT_MAX;
                    minC = INT_MAX;
                    for (k = 0; k < points.size(); k++) {
                        minR = min(minR, t[k].first);
                        minC = min(minC, t[k].second);
                    }
                    for (k = 0; k < points.size(); k++) {
                        t[k].first -= minR;
                        t[k].second -= minC;
                    }
                    sort(t.begin(), t.end());
                    shapeStr.clear();
                    for (auto p : t) {
                    //    printf ("(%d, %d)->",p.first, p.second);
                        shapeStr += "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
                    }
                    // cout << endl;
                    shapeStrColl.emplace_back(shapeStr);
                }
                for (k = 0; k < points.size(); k++) {
                    int tmp = points[k].first;
                    points[k].first = points[k].second;
                    points[k].second = tmp;
                }
                t = points;
                for (l = 0; l < 4; l++) {
                    for (k = 0; k < points.size(); k++) {
                        t[k].first = points[k].first * pontsTrans[l][0];
                        t[k].second = points[k].second * pontsTrans[l][1];
                    }
                    // 变换后的相对坐标
                    minR = INT_MAX;
                    minC = INT_MAX;
                    for (k = 0; k < points.size(); k++) {
                        minR = min(minR, t[k].first);
                        minC = min(minC, t[k].second);
                    }
                    for (k = 0; k < points.size(); k++) {
                        t[k].first -= minR;
                        t[k].second -= minC;
                    }
                    sort(t.begin(), t.end());
                    shapeStr.clear();
                    for (auto p : t) {
                    //    printf ("(%d, %d)->",p.first, p.second);
                        shapeStr += "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
                    }
                    // cout << endl;
                    shapeStrColl.emplace_back(shapeStr);
                }
                sort(shapeStrColl.begin(), shapeStrColl.end());
                if (shape.count(shapeStrColl[0]) == 0) {
                    shape.emplace(shapeStrColl[0]);
                }
            }
            
        }
    }
    return shape.size();
}


// LCP12
int minTime(vector<int>& time, int m)
{
    int i;
    int n = time.size();
    int t, cntDay, curMax;
    int left, right, mid;
    bool f = false;
    if (n <= m) {
        return 0;
    }
    right = 0;
    for (i = 0; i < n; i++) {
        // left = max(left, time[i]);
        right += time[i];
    }
    left = 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        t = 0;
        cntDay = 0;
        curMax = 0;
        f = false;
        for (i = 0; i < n; i++) {
            if (i == 3) {
                ;
            }
            if (t + time[i] > mid) {
                if (f) {
                    cntDay++;
                    t = time[i];
                    if (t > mid) {
                        // cntDay++;
                        t = 0;
                        f = true;
                        curMax = 0;
                        continue;
                    }
                    curMax = max(curMax, time[i]);
                    f = false;
                    continue;
                }
                if (time[i] >= curMax) {
                    f = true;
                    curMax = 0;
                } else {
                    f = true;
                    t -= curMax;
                    t += time[i];
                    curMax = 0;
                }

            } else {
                t += time[i];
                if (f == false) {
                    curMax = max(curMax, time[i]);
                }
            }
        }
        if (t > 0 || f) {
            cntDay++;
        }
        if (cntDay > m) {
            left = mid + 1;
        } else if (cntDay <= m) {
            right = mid - 1;
        }
    }
    return left;
}



// LCP45
vector<vector<int>> bicycleYard(vector<int>& position, vector<vector<int>>& terrain, vector<vector<int>>& obstacle)
{
    int i, k;
    int x, y, v, newV;
    int n = terrain.size();
    int m = terrain[0].size();
    int size;
    vector<int> pos;
    vector<vector<int>> ans;
    vector<vector<int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    set<vector<int>> visited;
    set<vector<int>> sameVPos;
    queue<vector<int>> q;
    vector<int> start = {position[0], position[1], 1, -1}; // x, y坐标, 速度, 方向

    q.push(start);
    // visited[position[0]][position[1]] = true;
    visited.insert({position[0], position[1], 1, 0});
    visited.insert({position[0], position[1], 1, 1});
    visited.insert({position[0], position[1], 1, 2});
    visited.insert({position[0], position[1], 1, 3});
    while (q.size()) {
        size = q.size();
        for (i = 0; i < size; i++) {
            pos = q.front();
            q.pop();
            for (k = 0; k < 4; k++) {
                x = pos[0] + directions[k][0];
                y = pos[1] + directions[k][1];
                v = pos[2];

                if (x < 0 || x >= n || y < 0 || y >= m) {
                    continue;
                }
                newV = v + terrain[pos[0]][pos[1]] - terrain[x][y] - obstacle[x][y];
                if (newV <= 0 || visited.count({x, y, newV, k}) == 1) {
                    continue;
                }
                visited.insert({x, y, newV, k});
                if (newV == 1) {
                    sameVPos.insert({x, y});
                }
                q.push({x, y, newV});
            }
        }
    }
    for (auto it : sameVPos) {
        ans.emplace_back(it);
    }
    return ans;
}


// LCP41
template<typename T>
void ClearQueue(queue<T>& q)
{
    queue<T> empty;
    swap(empty, q);
}
int CntChess(vector<string>& chessboard, int row, int col)
{
    int cnt, t;
    int i, j, k;
    int n = chessboard.size();
    int m = chessboard[0].size();
    vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    queue<pair<int, int>> q, tq;
    cnt = 0;
    for (k = 0; k < directions.size(); k++) {
        t = 0;
        i = row;
        j = col;
        while (1) {
            i += directions[k][0];
            j += directions[k][1];
            if (i < 0 || i >= n || j < 0 || j >= m) {
                ClearQueue<pair<int, int>>(q);
                break;
            }
            if (chessboard[i][j] == 'O') {
                q.push({i, j});
            } else if (chessboard[i][j] == 'X') {
                if (q.size() == 0) {
                    break;
                }
                t += q.size();
                while (!q.empty()) {
                    auto p = q.front();
                    tq.push(p);
                    chessboard[p.first][p.second] = 'X';
                    q.pop();
                }
                while (!tq.empty()) {
                    auto p = tq.front();
                    t += CntChess(chessboard, p.first, p.second);
                    tq.pop();
                }
                break;
            } else if (chessboard[i][j] == '.') {
                ClearQueue<pair<int, int>>(q);
                break;
            }
        }
        cnt += t;
    }
    return cnt;
}
int flipChess(vector<string>& chessboard)
{
    int i, j;
    int ans;
    int n = chessboard.size();
    int m = chessboard[0].size();
    vector<string> tmpChess;

    ans = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (chessboard[i][j] == '.') {
                tmpChess = chessboard;
                tmpChess[i][j] = 'X';
                ans = max(ans, CntChess(tmpChess, i, j));
            }
        }
    }
    return ans;
}


// LC300
int lengthOfLIS(vector<int> &nums)
{
    int i, j;
    int n = nums.size();
    int ans;
    vector<int> dp(n, 1); // 以i下标结尾的最长递增子序列长度

    for (i = 1; i < n; i++) {
        for (j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    ans = dp[0];
    for (i = 1; i < n; i++) {
        ans = max(ans, dp[i]);
    }
    return ans;
}
// LC673
int findNumberOfLIS(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    vector<int> t;
    int curLen, cnt;
    vector<pair<int, int>> dp(n, {0, 0}); // 以i下标结尾的最长递增子序列长度, 以及对应的序列个数

    dp[0] = {1, 1};
    for (i = 1; i < n; i++) {
        cnt = curLen = 0;
        for (j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) {
                if (dp[j].first + 1 > curLen) {
                    curLen = dp[j].first + 1;
                    cnt = dp[j].second;
                } else if (dp[j].first + 1 == curLen) {
                    cnt += dp[j].second;
                }
            }
        }
        if (curLen == 0) {
            dp[i] = {1, 1};
        } else {
            dp[i] = {curLen, cnt};
        }
    }
    int maxLen = 0;
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (dp[i].first > maxLen) {
            maxLen = dp[i].first;
            ans = dp[i].second;
        } else if (dp[i].first == maxLen) {
            ans += dp[i].second;
        }
    }
    return ans;
}


// LC1125
void DFSFindSmallestSufficientTeam(vector<string>& req_skills, int curIdx, unordered_map<string, vector<int>>& skillsPeopleKnow,
    unordered_map<int, unordered_set<string>>& peopleSkills, vector<int>& record, vector<int>& ans, int& minTeamSize)
{
    if (curIdx == req_skills.size()) {
        if (minTeamSize == -1 || record.size() < minTeamSize) {
            minTeamSize = record.size();
            ans = record;
        }
        return;
    }
    int i;
    bool f = false;
    // 判断这个技能是否已经有人在前面会了
    for (auto r : record) {
        if (peopleSkills[r].count(req_skills[curIdx]) == 1) {
            f = true;
            break;
        }
    }
    if (f) {
        DFSFindSmallestSufficientTeam(req_skills, curIdx + 1, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);
    } else {
        for (auto it : skillsPeopleKnow[req_skills[curIdx]]) {  // 这些人会req_skills[curIdx]
            record.emplace_back(it);
            if (minTeamSize != -1 && record.size() > minTeamSize) {
                record.pop_back();
                return;
            }
            DFSFindSmallestSufficientTeam(req_skills, curIdx + 1, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);
            record.pop_back();
        }
    }
}
vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people)
{
    int minTeamSize;
    unordered_map<string, vector<int>> skillsPeopleKnow; // 技能 - 会的人物列表
    unordered_map<int, unordered_set<string>> peopleSkills; // 人物 - 会的技能列表
    vector<int> ans;
    vector<int> record;
    vector<vector<int>> results;

    int i, j;
    int num = people.size();
    for (i = 0; i < num; i++) {
        for (j = 0; j < people[i].size(); j++) {
            skillsPeopleKnow[people[i][j]].emplace_back(i);
            peopleSkills[i].emplace(people[i][j]);
        }
    }
    minTeamSize = -1;
    DFSFindSmallestSufficientTeam(req_skills, 0, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);

    return ans;
}


// LC1041
bool CheckIfEqual(int len, vector<vector<int>>& source, int startIdx)
{
    int n = source.size();
    int i, j;

    for (i = 0; i < len; i++) {
        if (i + startIdx >= n) {
            break;
        }
        if (source[i + startIdx] != source[i]) {
            return false;
        }
    }
    return true;
}
bool isRobotBounded(string instructions)
{
    int i, k;
    int n = instructions.size();
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int curD, x, y;
    bool loop = false;
    vector<vector<int>> pos;
    vector<int> curPos;
    vector<int> t;

    pos.push_back({0, 0, 0});
    curD = 0;
    curPos = {0, 0};
    for (k = 0; k < 10; k++) {
        for (i = 0; i < n; i++) {
            if (instructions[i] == 'G') {
                x = curPos[0] + directions[curD][0];
                y = curPos[1] + directions[curD][1];
                t = {x, y, curD};
                curPos = {x, y};
                pos.emplace_back(t);
            } else if (instructions[i] == 'L') {
                curD = (curD + 3) % 4;
                t = {curPos[0], curPos[1], curD};
                pos.emplace_back(t);
            } else {
                curD = (curD + 1) % 4;
                t = {curPos[0], curPos[1], curD};
                pos.emplace_back(t);
            }
        }
    }
    /*for (auto p : pos) {
        for (auto a : p) {
            cout << a << " ";
        }
        cout << endl;
    } */
    n = pos.size();
    int len = 1;
    while (1) {
        loop = true;
        i = len;
        while (1) {
            if (CheckIfEqual(len, pos, i) == false) {
                loop = false;
                break;
            }
            i += len;
            if (i >= n) {
                loop = true;
                break;
            }
        }
        if (loop) {
            return true;
        }
        len++;
        if (len * 2 > n) {
            break;
        }
    }
    return false;
}


// LC2560
// 求k个不相邻最大值中的最小值
int minCapability(vector<int>& nums, int k)
{
    int i;
    int size = nums.size();
    if (size == 1) {
        return nums[0];
    } else if (size == 2) {
        return min(nums[0], nums[1]);
    }

    int left, right, mid;
    int curMaxSteal;
    vector<int> dp(size, 0);
    left = INT_MAX;
    right = INT_MIN;
    for (auto n : nums) {
        left = min(left, n);
        right = max(right, n);
    }
    // dp[i] = max(dp[i - 1], dp[i - 2] + 1) 下标i选择窃取或不窃取能得手的最大房间数
    while (left <= right) {
        mid = (right - left) / 2 + left;  // 可能的最大窃取金额
        curMaxSteal = 0;
        if (nums[0] <= mid) {
            dp[0] = 1;
            curMaxSteal = 1;
        } else {
            dp[0] = 0;
        }
        if (nums[1] <= mid) {
            dp[1] = 1;
            curMaxSteal = 1;
        } else {
            dp[1] = dp[0];
        }
        for (i = 2; i < size; i++) {
            if (nums[i] <= mid) {
                dp[i] = max(dp[i - 1], dp[i - 2] + 1);
                curMaxSteal = max(curMaxSteal, dp[i]);
            } else {
                dp[i] = dp[i - 1];
            }
        }
        if (curMaxSteal >= k) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2616
// 求p个不相邻最小值中的最大值,与上题相反
int minimizeMax(vector<int>& nums, int p)
{
    int i;
    int n = nums.size();
    vector<int> diff;

    if (p == 0) {
        return 0;
    }
    sort(nums.begin(), nums.end());
    for (i = 1; i < n; i++) {
        diff.emplace_back(nums[i] - nums[i - 1]);
    }
    n = diff.size();
    if (n == 1) {
        return diff[0];
    } else if (n == 2) {
        return min(diff[0], diff[1]);
    }

    vector<int> dp(n, 0);
    int left, right, mid;
    int curMinCnt;

    left = INT_MAX;
    right = INT_MIN;
    for (auto d : diff) {
        left = min(d, left);
        right = max(d, right);
    }
    while (left <= right) {
        mid = (right - left) / 2 + left; // mid 可能的最大值
        curMinCnt = 0;
        if (diff[0] <= mid) {
            dp[0] = 1;
            curMinCnt = 1;
        } else {
            dp[0] = 0;
        }

        if (diff[1] <= mid) {
            dp[1] = 1;
            curMinCnt = 1;
        } else {
            dp[1] = dp[0];
        }

        for (i = 2; i < n; i++) {
            if (diff[i] <= mid) {
                dp[i] = max(dp[i - 1], dp[i - 2] + 1);
                curMinCnt = max(curMinCnt, dp[i]);
            } else {
                dp[i] = dp[i - 1];
            }
        }
        if (curMinCnt >= p) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2608
int BFSFindMinLoopSize(int n, unordered_map<int, unordered_set<int>>& edges, int curNode)
{
    int d, val, loopSize, minLoopSize;
    int i, k;
    queue<pair<int, int>> q; // curNode - fromNode
    pair<int, int> t;
    vector<bool> visited(n, false);
    vector<int> distance(n);

    q.push({curNode, - 1});
    d = 0;
    minLoopSize = INT_MAX;
    while (q.size()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            t = q.front();
            q.pop();
            visited[t.first] = true;
            distance[t.first] = d;
            for (auto it : edges[t.first]) {
                if (it != t.second && visited[it] == true) {
                    loopSize = distance[it] + distance[t.first] + 1;
                    // printf ("n = %d, loopSize = %d\n", curNode, loopSize);
                    minLoopSize = min(minLoopSize, loopSize);
                    // return loopSize; 不能直接退出
                }
                if (it != t.second && visited[it] == false) {
                    q.push({it, t.first});
                }
            }
        }
        d++;
    }
    return minLoopSize == INT_MAX ? -1 : minLoopSize;
}
int findShortestCycle(int n, vector<vector<int>>& edges)
{
    unsigned int i;
    int loopSize, ans;
    unordered_map<int, unordered_set<int>> e;
    for (i = 0; i < edges.size(); i++) {
        e[edges[i][0]].emplace(edges[i][1]);
        e[edges[i][1]].emplace(edges[i][0]);
    }

    ans = INT_MAX;
    for (i = 0; i < n; i++) {
        loopSize = BFSFindMinLoopSize(n, e, i);
        if (loopSize > 0) {
            ans = min(ans, loopSize);
        }
    }
    return ans == INT_MAX ? -1 : ans;
}


// LC2064
int minimizedMaximum(int n, vector<int>& quantities)
{
    int sum;
    int left, right, mid;

    left = 1;
    right = quantities[0];
    for (auto q : quantities) {
        right = max(right, q);
    }

    while (left <= right) {
        mid = (right - left) / 2 + left;
        sum = 0;
        for (auto q : quantities) {
            sum += static_cast<int>(ceil(q * 1.0 / mid));
        }
        if (sum > n) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1042
vector<int> gardenNoAdj(int n, vector<vector<int>>& paths)
{
    int i, j;
    int curVal;
    vector<int> ans(n, 0);
    unordered_map<int, int> gardenVal;
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> s;
    bool comflict = false;

    for (auto p : paths) {
        edges[p[0]].emplace(p[1]);
        edges[p[1]].emplace(p[0]);
    }
    for (i = 1; i <= n; i++) {
        if (gardenVal.count(i) == 0) {
            gardenVal[i] = 1;
            for (auto e : edges[i]) {
                if (gardenVal.count(e) == 0) {
                    s.clear();
                    for (auto it : edges[e]) {
                        if (gardenVal.count(it) == 1) {
                            s.emplace(gardenVal[it]);
                        }
                    }
                    for (j = 1; j <= 4; j++) {
                        if (s.count(j) == 0) {
                            gardenVal[e] = j;
                            break;
                        }
                    }
                }
            }
        } else {
            s.clear();
            for (auto e : edges[i]) {
                if (gardenVal.count(e) == 1) {
                    s.emplace(gardenVal[e]);
                }
            }
            comflict = false;
            for (auto it : s) {
                if (it == gardenVal[i]) {
                    comflict = true;
                    break;
                }
            }
            if (comflict) {
                for (j = 1; j <= 4; j++) {
                    if (s.count(j) == 0) {
                        gardenVal[i] = j;
                        break;
                    }
                }
            }
        }
    }
    for (auto it : gardenVal) {
        ans[it.first - 1] = it.second;
    }
    return ans;
}
vector<int> gardenNoAdj_1(int n, vector<vector<int>>& paths)
{
    int i, j;
    int curVal;
    vector<int> ans(n, 0);
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> s;

    for (auto p : paths) {
        edges[p[0]].emplace(p[1]);
        edges[p[1]].emplace(p[0]);
    }
    for (i = 1; i <= n; i++) {
        s.clear();
        for (auto e : edges[i]) {
            if (ans[e - 1] != 0) {
                s.emplace(ans[e - 1]);
            }
        }
        for (j = 1; j <= 4; j++) {
            if (s.count(j) == 0) {
                ans[i - 1] = j;
                break;
            }
        }
    }
    return ans;
}


// LC2641
TreeNode* replaceValueInTree(TreeNode* root)
{
    int i, j;
    int n;
    queue<pair<TreeNode *, TreeNode *>> q; // pair 子 - 父
    vector<vector<pair<TreeNode *, TreeNode *>>> treeLayerLists;
    vector<pair<TreeNode *, TreeNode *>> v;
    pair<TreeNode *, TreeNode *> t;

    q.push({root, nullptr});
    while (q.size()) {
        n = q.size();
        v.clear();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            v.emplace_back(t);
            if (t.first->left != nullptr) {
                q.push({t.first->left, t.first});
            }
            if (t.first->right != nullptr) {
                q.push({t.first->right, t.first});
            }
        }
        treeLayerLists.emplace_back(v);
    }
    unordered_map<TreeNode *, int> fatherSum;
    n = treeLayerLists.size();
    vector<int> layerSum(n, 0);
    for (i = 0; i < n; i++) {
        for (j = 0; j < treeLayerLists[i].size(); j++) {
            fatherSum[treeLayerLists[i][j].second] += treeLayerLists[i][j].first->val;
            layerSum[i] += treeLayerLists[i][j].first->val;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < treeLayerLists[i].size(); j++) {
            treeLayerLists[i][j].first->val = layerSum[i] - fatherSum[treeLayerLists[i][j].second];
        }
    }
    return root;
}


// LC2642
class Graph {
public:
    int n;
    unordered_map<int, set<vector<int>>> edges;
    Graph(int n, vector<vector<int>>& edges)
    {
        for (auto e : edges) {
            (this->edges)[e[0]].insert({e[1], e[2]});
        }
        this->n = n;
    }
    
    void addEdge(vector<int> edge)
    {
        (this->edges)[edge[0]].insert({edge[1], edge[2]});
    }
    
    int shortestPath(int node1, int node2) // 超时
    {
        pair<int, int> t;
        vector<int> nodeDist(n, INT_MAX);
        queue<pair<int, int>> q;

        if (node1 == node2) {
            return 0;
        }
        q.push({node1, 0});
        while (q.size()) {
            t = q.front();
            q.pop();
            if (nodeDist[t.first] < t.second) {
                continue;
            }
            nodeDist[t.first] = t.second;
            for (auto e : edges[t.first]) {
                if (nodeDist[e[0]] > t.second + e[1]) {
                    nodeDist[e[0]] = t.second + e[1];
                    q.push({e[0], t.second + e[1]});
                }
            }
        }
        return nodeDist[node2] == INT_MAX ? -1 : nodeDist[node2];
    }
};

class Graph1 {
public:
    int n;
    vector<vector<pair<int, int>>> edges;
    Graph1(int n, vector<vector<int>>& edges)
    {
        this->n = n;
        this->edges = vector<vector<pair<int, int>>>(n);
        for (auto e : edges) {
            (this->edges)[e[0]].push_back({e[1], e[2]});
        }
    }
    
    void addEdge(vector<int> edge)
    {
        (this->edges)[edge[0]].push_back({edge[1], edge[2]});
    }
    
    int shortestPath(int node1, int node2)
    {
        pair<int, int> t;
        vector<int> nodeDist(n, INT_MAX);
        queue<pair<int, int>> q;
        if (node1 == node2) {
            return 0;
        }
        q.push({node1, 0});
        while (q.size()) {
            t = q.front();
            q.pop();
            if (nodeDist[t.first] < t.second) {
                continue;
            }
            nodeDist[t.first] = t.second;
            for (auto e : edges[t.first]) {
                if (nodeDist[e.first] > t.second + e.second) {
                    nodeDist[e.first] = t.second + e.second;
                    q.push({e.first, t.second + e.second});
                }
            }
        }
        return nodeDist[node2] == INT_MAX ? -1 : nodeDist[node2];
    }
};

// LC2646
unordered_map<int, int> tripCostData;
void DFSCalcTripCost(unordered_map<int, unordered_set<int>>& e, int from, int cur, int dest, string route, vector<int>& price, bool& find)
{
    if (find) {
        return;
    }
    if (cur == dest) {
        route[route.size() - 1] = '\0';
        
        int i;
        int node;
        int t;
        vector<string> vs = MySplit(route, '_');
        for (i = 0; i < vs.size(); i++) {
            node = atoi(vs[i].c_str());
            tripCostData[node] += price[node];
        }
        find = true;
        return;
    }
    for (auto it : e[cur]) {
        if (it != from) {
            DFSCalcTripCost(e, cur, it, dest, route + to_string(it) + "_", price, find);
        }
    }
}
int CalcMaxLessCost(unordered_map<int, unordered_set<int>>& edges, int cur, int from, bool chosen,
    unordered_map<int, int>& tripCostData, map<pair<int, bool>, int>& lessCost)
{
    if (edges[cur].size() == 1 && *edges[cur].begin() == from) {
        if (chosen) {
            lessCost[{cur, chosen}] = tripCostData[cur] / 2;
        } else {
            lessCost[{cur, chosen}] = 0;
        }
        return lessCost[{cur, chosen}];
    }
    for (auto it : edges[cur]) {
        if (it != from) {
            if (lessCost.count({it, false}) == 0) {
                lessCost[{it, false}] = CalcMaxLessCost(edges, it, cur, false, tripCostData, lessCost);
            }
            if (lessCost.count({it, true}) == 0) {
                lessCost[{it, true}] = CalcMaxLessCost(edges, it, cur, true, tripCostData, lessCost);
            }
        }
    }
    int ans = 0;
    if (chosen) {
        ans = tripCostData[cur] / 2;
        for (auto it : edges[cur]) {
            if (it != from) {
                ans += lessCost[{it, false}];
            }
        }
        lessCost[{cur, chosen}] = ans;
    } else {
        for (auto it : edges[cur]) {
            if (it != from) {
                ans += max(lessCost[{it, false}], lessCost[{it, true}]);
            }
        }
        lessCost[{cur, chosen}] = ans;
    }
    return ans;
}
int minimumTotalPrice(int n, vector<vector<int>>& edges, vector<int>& price, vector<vector<int>>& trips)
{
    int ans;
    unordered_map<int, unordered_set<int>> e;
    unordered_set<int> curChosenNode;
    bool find = false;

    if (edges.size() == 0) {
        ans = 0;
        for (auto trip : trips) {
            ans += price[trip[0]] / 2;
        }
        return ans;
    }
    for (auto edge : edges) {
        e[edge[0]].emplace(edge[1]);
        e[edge[1]].emplace(edge[0]);
    }
    string route;
    for (auto trip : trips) {
        if (trip[0] == trip[1]) {
            tripCostData[trip[0]] += price[trip[0]];
            continue;
        }
        find = false;
        route.clear();
        route = to_string(trip[0]) + "_";
        DFSCalcTripCost(e, -1, trip[0], trip[1], route, price, find);
    }

    ans = 0;
    for (auto it : tripCostData) {
        // printf ("%d : %d\n", it.first, it.second);
        ans += it.second;
    }
    // printf ("ans = %d\n", ans);
    int startNode = tripCostData.begin()->first;
    map<pair<int, bool>, int> lessCost; // <node - 是否减半> - 所节约的最大费用
    int a = CalcMaxLessCost(e, startNode, -1, true, tripCostData, lessCost); // 从startNode出发, 该点选择减半
    int b = CalcMaxLessCost(e, startNode, -1, false, tripCostData, lessCost); // 从startNode出发, 该点选择不减半
    return ans - max(a, b);
}


// LC1733
int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships)
{
    int i;
    int m = languages.size();
    int popularPerson, lessPopularPerson;
    unordered_map<int, unordered_set<int>> skill;
    unordered_map<int, int> langCnt;
    for (i = 0; i < m; i++) {
        for (auto lang : languages[i]) {
            skill[i + 1].emplace(lang);
        }
    }
    int sk;
    unordered_set<int> cantCommunicate;
    for (auto fr : friendships) {
        unordered_set<int>::iterator it;
        for (it = skill[fr[0]].begin(); it != skill[fr[0]].end(); it++) {
            if (skill[fr[1]].count(*it) == 1) {
                break; 
            } else {
                sk = *it;
            }
        }
        if (it == skill[fr[0]].end()) {
            cantCommunicate.emplace(fr[0]);
            cantCommunicate.emplace(fr[1]);
        }
    }
    for (auto it : cantCommunicate) {
        for (auto lang : skill[it]) {
            langCnt[lang]++;
        }
    }
    int t = 0;
    int lang = 0;
    for (auto it : langCnt) {
        if (t < it.second) {
            t = it.second;
            lang = it.first;
        }
    }
    int ans = 0;
    for (auto it : cantCommunicate) {
        if (skill[it].count(lang) == 0) {
            ans++;
        }
    }
    return ans;
}


// 投n个6面骰子概率分布
vector<double> dicesProbability(int n)
{
    // p[n][sum] - 投n个骰子和为sum的概率
    vector<vector<double>> p(n + 1, vector<double>(6 * n + 1, 0.0));
    int i, j, k;
    for (i = 1; i <= 6; i++) {
        p[1][i] = 1.0 / 6;
    }
    for (i = 2; i <= n; i++) {
        for (j = i - 1; j <= 6 * (i - 1); j++) {
            for (k = 1; k <= 6; k++) {
                p[i][j + k] += p[i - 1][j] * (1.0 / 6);
            }
        }
    }
    vector<double> ans;
    for (i = n; i <= n * 6; i++) {
        ans.emplace_back(p[n][i]);
    }
    return ans;
}


// LC1043
int maxSumAfterPartitioning(vector<int>& arr, int k)
{
    // dp[i][j] - 以i为下标长度为j的子数组最大和, j <= i + 1, j >= 0, j <= k
    int i, j, p;
    int t, len;
    int n = arr.size();
    vector<vector<int>> dp(n, vector<int>(k + 1, 0));

    // 从后往前找以i为下标,长度为k的子数组中的最大值
    vector<vector<int>> rangeMax(n, vector<int>(n, 0));
    for (i = n - 1; i >= 0; i--) {
        t = 0;
        for (j = i; j >= 0; j--) {
            if (j + k == i) {
                break;
            }
            t = max(t, arr[j]);
            rangeMax[j][i] = t;
        }
    }
    dp[0][1] = arr[0];
    for (i = 1; i < n; i++) {
        for (j = i; j >= 0; j--) {
            if (j + k == i) {
                break;
            }
            len = i - j + 1;
            dp[i][len] = rangeMax[j][i] * len;
            t = 0;
            for (p = 1; p <= k; p++) {
                if (i - len + 1 - p < 0) {
                    break;
                }
                t = max(t, dp[i - len][p]);
            }
            dp[i][i - j + 1] += t;
        }
    }
    int ans = 0;

    for (p = 1; p <= k; p++) {
        ans = max(ans, dp[n - 1][p]);
    }
    return ans;
}


// LC2320
int countHousePlacements(int n)
{
    // dp[i][0] - 下标i处不放置房子方案数; dp[i][1] - 下标i处放置房子方案数; 下标从1开始
    // dp[i][1] = dp[i - 1][0]; dp[i][0] = dp[i - 1][0] + dp[i - 1][1];
    int i;
    int mod = 1000000007;
    long long ans, t;
    vector<vector<long long>> dp(n + 1, vector<long long>(2, 0));

    dp[1][0] = 1;
    dp[1][1] = 1;
    for (i = 2; i <= n; i++) {
        dp[i][1] = dp[i - 1][0];
        dp[i][0] = (dp[i - 1][0] + dp[i - 1][1]) % mod;
    }
    t = dp[n][0] + dp[n][1];
    ans = t * t % mod;
    return ans;
}


// LC1178
void DFSGetPuzzlesHash(string& t, int idx, string& ori, unordered_map<string, int>& wordsHashCnt,
    unordered_map<string, int>& puzzlesCnt)
{
    int i;
    if (idx == ori.size()) {
        if (wordsHashCnt.count(t) == 1) {
            puzzlesCnt[ori] += wordsHashCnt[t];
        }
        return;
    }
    for (i = 0; i <= 1; i++) {
        t[ori[idx] - 'a'] = i + '0';
        DFSGetPuzzlesHash(t, idx + 1, ori, wordsHashCnt, puzzlesCnt);
    }
}
vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles)
{
    int i;
    unordered_set<char> wordHash;
    unordered_map<string, int> puzzlesCnt;
    unordered_set<string> uniquePuzzlesSet;
    unordered_map<string, int> wordsHashCnt;
    unordered_map<string, string> uniqueWordsHashMap;
    string t(26, '0');
    int n = puzzles.size();
    vector<int> ans(n, 0);

    for (auto word : words) {
        if (uniqueWordsHashMap.count(word) == 1) {
            wordsHashCnt[uniqueWordsHashMap[word]]++;
            continue;
        }
        wordHash.clear();
        t.assign(26, '0');
        for (i = 0; i < word.size(); i++) {
            t[word[i] - 'a'] = '1';
            wordHash.emplace(word[i]);
            if (wordHash.size() > 7) {
                break;
            }
        }
        if (i == word.size()) {
            wordsHashCnt[t]++;
            uniqueWordsHashMap[word] = t;
        }
    }
    for (auto p : puzzles) {
        if (uniquePuzzlesSet.count(p) == 1) {
            continue;
        }
        uniquePuzzlesSet.emplace(p);
        t.assign(26, '0');
        t[p[0] - 'a'] = '1';
        DFSGetPuzzlesHash(t, 1, p, wordsHashCnt, puzzlesCnt);
        // cout << usedPuzzlesHash.size() << endl;
    }
    for (i = 0; i < n; i++) {
        ans[i] = puzzlesCnt[puzzles[i]];
    }
    return ans;
}


// LCP72
vector<int> supplyWagon(vector<int>& supplies)
{
    int i;
    int n;
    vector<int> t, v;
    pair<int, int> p;
    vector<pair<int, int>> vp;

    v = supplies;
    while (v.size() > supplies.size() / 2) {
        t.clear();
        n = v.size();
        for (i = 1; i < n; i++) {
            vp.push_back({v[i] + v[i - 1], i - 1});
        }
        sort (vp.begin(), vp.end());
        p = vp[0];
        vp.clear();
        for (i = 0; i < n; i++) {
            if (i == p.second) {
                t.emplace_back(p.first);
                i++;
            } else {
                t.emplace_back(v[i]);
            }
        }
        v = t;
    }
    return v;
}


// LCP73
vector<string> MySplit(string& s, string separate)
{
    int idx;
    string t;
    vector<string> ans;

    t = s;
    while (1) {
        idx = t.find(separate);
        if (idx == string::npos) {
            ans.emplace_back(t);
            break;
        }
        ans.emplace_back(t.substr(0, idx));
        t = t.substr(idx + separate.size());
    }
    return ans;
}
int adventureCamp(vector<string>& expeditions)
{
    int i;
    int ans, newFound;
    int n = expeditions.size();
    vector<int> newFoundSite(n, 0);
    unordered_set<string> visitedSite;
    vector<string> vs;

    if (expeditions[0].size() > 0) {
        vs = MySplit(expeditions[0], "->");
        for (auto t : vs) {
            visitedSite.emplace(t);
        }
    }
    for (i = 1; i < n; i++) {
        if (expeditions[i].size() == 0) {
            continue;
        }
        vs = MySplit(expeditions[i], "->");
        for (auto t : vs) {
            if (visitedSite.count(t) == 0) {
                newFoundSite[i]++;
                visitedSite.emplace(t);
            }
        }
    }
    ans = -1;
    newFound = 0;
    for (i = 1; i < n; i++) {
        if (newFoundSite[i] > newFound) {
            newFound = newFoundSite[i];
            ans = i;
        }
    }
    return ans;
}


// LCP74 有问题
// [x1, y1, x2, y2] [x3, y3, x4, y4]
// (x4 - x1) * (x3 - x2) <= 0 且 (y4 - y1) * (y3 - y2) <= 0
bool CheckIfCoincided(vector<double>& a, vector<double>& b)
{
    return ((b[2] - a[0]) * (b[0] - a[2])) <= 0 && ((b[3] - a[1]) * (b[1] - a[3]) <= 0);
}
int fieldOfGreatestBlessing(vector<vector<int>>& forceField)
{
    int i, j, k, l;
    int m, cnt, ans;
    // 转换forceFiel为左下-右上坐标
    vector<vector<double>> newForceField;
    for (auto t : forceField) {
        newForceField.push_back({t[0] - t[2] / 2.0, t[1] - t[2] / 2.0, t[0] + t[2] / 2.0, t[1] + t[2] / 2.0});
    }
    /*for (auto t : newForceField) {
        for (auto v : t) {
            cout << v << " ";
        }
        cout << endl;
    }*/

    vector<vector<double>> coincidedTriange;
    ans = 1;
    m = newForceField.size();
    for (i = 0; i < m; i++) {
        coincidedTriange.clear();
        coincidedTriange.emplace_back(newForceField[i]);
        for (j = i + 1; j < m; j++) {
            // [x1, y1, x2, y2] [x3, y3, x4, y4]
            // (x4 - x1) * (x3 - x2) <= 0 且 (y4 - y1) * (y3 - y2) <= 0
            /*if ((newForceField[j][2] - newForceField[i][0]) * (newForceField[j][0] - newForceField[i][2]) <= 0 &&
                (newForceField[j][3] - newForceField[i][1]) * (newForceField[j][1] - newForceField[i][3]) <= 0)
            {
                cnt++;
            }*/
            // coincidedTriange.clear();
            if (CheckIfCoincided(newForceField[i], newForceField[j])) {
                // coincidedTriange.emplace_back(newForceField[i]);
                coincidedTriange.emplace_back(newForceField[j]);
                for (k = j + 1; k < m; k++) {
                    for (l = 0; l < coincidedTriange.size(); l++) {
                        if (CheckIfCoincided(newForceField[l], newForceField[k]) == false) {
                            break;
                        }
                    }
                    if (l == coincidedTriange.size()) {
                        coincidedTriange.emplace_back(newForceField[k]);
                    }
                }
                ans = max(ans, static_cast<int>(coincidedTriange.size()));
            }
        }
    }
    return ans;
}

void DFSGetMaxCoincided(vector<vector<double>>& forceField, int curIdx, vector<vector<double>>& coincidedTriange, int cnt, int& maxNum)
{
    int i, k;
    int n = forceField.size();

    if (curIdx == n) {
        maxNum = max(cnt, maxNum);
        return;
    }
    for (i = curIdx; i < n; i++) {
        if (coincidedTriange.size() == 0) {
            coincidedTriange.emplace_back(forceField[i]);
            DFSGetMaxCoincided(forceField, i + 1, coincidedTriange, cnt + 1, maxNum);
            coincidedTriange.pop_back();
        } else {
            for (k = 0; k < coincidedTriange.size(); k++) {
                if (CheckIfCoincided(forceField[k], forceField[curIdx]) == false) {
                    break;
                }
            }
            if (k == coincidedTriange.size()) {
                coincidedTriange.emplace_back(forceField[curIdx]);
                DFSGetMaxCoincided(forceField, i + 1, coincidedTriange, cnt + 1, maxNum);
                coincidedTriange.pop_back();
            } else {
                DFSGetMaxCoincided(forceField, i + 1, coincidedTriange, cnt, maxNum);
            }
        }
    }
}
int fieldOfGreatestBlessing1(vector<vector<int>>& forceField)
{
    int i, j, k, l;
    int m, cnt, ans;
    // 转换forceFiel为左下-右上坐标
    vector<vector<double>> newForceField;
    for (auto t : forceField) {
        newForceField.push_back({t[0] - t[2] / 2.0, t[1] - t[2] / 2.0, t[0] + t[2] / 2.0, t[1] + t[2] / 2.0});
    }
    /*for (auto t : newForceField) {
        for (auto v : t) {
            cout << v << " ";
        }
        cout << endl;
    }*/

    vector<vector<double>> coincidedTriange;
    ans = 1;
    m = newForceField.size();
    for (i = 0; i < m; i++) {
        coincidedTriange.clear();
        coincidedTriange.emplace_back(newForceField[i]);
        for (j = i + 1; j < m; j++) {
            for (k = 0; k < coincidedTriange.size(); k++) {
                if (CheckIfCoincided(newForceField[k], newForceField[j]) == false) {
                    break;
                }
            }
            if (k == coincidedTriange.size()) {
                coincidedTriange.emplace_back(newForceField[j]);
            }
        }
        ans = max(ans, static_cast<int>(coincidedTriange.size()));
    }
    return ans;
}


// LC2653
vector<int> getSubarrayBeauty(vector<int>& nums, int k, int x)
{
    int i;
    int n = nums.size();
    int t;
    vector<int> ans;
    map<int, int> cnt;
    for (i = 0; i < k; i++) {
        cnt[nums[i]]++;
    }
    t = 0;
    for (auto it : cnt) {
        t += it.second;
        if (t >= x) {
            ans.emplace_back(it.first < 0 ? it.first : 0);
            break;
        }
    }
    for (i = k; i < n; i++) {
        cnt[nums[i]]++;
        if (cnt[nums[i - k]] == 1) {
            cnt.erase(nums[i - k]);
        } else {
            cnt[nums[i - k]]--;
        }
        t = 0;
        for (auto it : cnt) {
            t += it.second;
            if (t >= x) {
                ans.emplace_back(it.first < 0 ? it.first : 0);
                break;
            }
        }
    }
    return ans;
}

// LC241 有问题
void DivideExpression(string& expression, vector<int>& nums, vector<char>& signs)
{
    int i;
    int curIdx;
    int n = expression.size();
    unordered_set<char> sign = {'+', '-', '*'};

    curIdx = 0;
    for (i = 0; i < n; i++) {
        if (sign.count(expression[i]) == 1) {
            signs.emplace_back(expression[i]);
            nums.emplace_back(atoi(expression.substr(curIdx, i - curIdx).c_str()));
            curIdx = i + 1;
        }
    }
    nums.emplace_back(atoi(expression.substr(curIdx).c_str()));
}
void BFAllPossibleExpression(int n, vector<int>& record, vector<bool>& visited, 
    vector<tuple<char, int , int>>& expressionData, set<vector<int>>& possibleAns)
{
    int i;
    int no, t, a, b;
    char sign;
    vector<int> middleAns;
    if (record.size() == n) {
        middleAns.clear();
        auto tmpExpressionData = expressionData;
        for (i = 0; i < record.size(); i++) {
            no = record[i];
            sign = get<0>(tmpExpressionData[no]);
            a = get<1>(tmpExpressionData[no]);
            b = get<2>(tmpExpressionData[no]);
            if (sign == '+') {
                t = a + b;
            } else if (sign == '-') {
                t = a - b;
            } else {
                t = a * b;
            }
            middleAns.emplace_back(t);
            if (no >= 1) {
                get<2>(tmpExpressionData[no - 1]) = t;
            }
            if (no < n - 1) {
                get<1>(tmpExpressionData[no + 1]) = t;
            }
        }
        possibleAns.emplace(middleAns);
        return;
    }
    for (i = 0; i < n ; i++) {
        if (visited[i] == false) {
            visited[i] = true;
            record.emplace_back(i);
            BFAllPossibleExpression(n, record, visited, expressionData, possibleAns);
            visited[i] = false;
            record.pop_back();
        }
    }
}
vector<int> diffWaysToCompute(string expression)
{
    int i;
    vector<int> ans;
    vector<int> nums;
    vector<char> signs;

    DivideExpression(expression, nums, signs);
    if (signs.size() == 0) {
        return nums;
    }
    vector<tuple<char, int , int>> expressionData;
    for (i = 0; i < signs.size(); i++) {
        expressionData.push_back({signs[i], nums[i], nums[i + 1]});
    }
    // 根据signs大小枚举所有运算顺序, 共 (signs.size())!种
    int n = signs.size();
    set<vector<int>> possibleAns;
    vector<int> record;
    vector<bool> visited(n, false);
    BFAllPossibleExpression(n, record, visited, expressionData, possibleAns);
    for (auto it : possibleAns) {
        ans.emplace_back(it[it.size() - 1]);
    }
    return ans;
}


// LC2439
int minimizeArrayValue(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left, right, mid;
    long long diff;

    left = right = nums[0];
    for (i = 0; i < n; i++) {
        left = min(left, nums[i]);
        right = max(right, nums[i]);
    }
    while (left <= right) {
        mid = (right - left) / 2 + left;
        diff = 0;
        for (i = 0; i < n; i++) {
            diff += static_cast<long long>(mid) - nums[i];
            if (diff < 0) {
                break;
            }
        }
        if (diff < 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right + 1;
}


// LC935
int knightDialer(int n)
{
    // dp[n][x] - 走n步落在x的总情况
    // 1 - (6, 8) 2 - (7, 9) 3 - (4, 8) 4 - （0, 3, 9）
    // 5 - () 6 - (0, 1, 7) 7 - (2, 6)
    // 8 - (1, 3) 9 - (2, 4) 0 - (4, 6)
    // dp[n][x] = dp[n - 1][a] + dp[n - 1][b] + ..., 所求 sum(dp[n])
    int i;
    int mod = 1000000007;
    vector<vector<long long>> dp(n + 1, vector<long long>(10, 0));
    
    for (i = 0; i < 10; i++) {
        dp[1][i] = 1;
    }
    for (i = 2; i <= n; i++) {
        dp[i][1] = (dp[i - 1][6] + dp[i - 1][8]) % mod;
        dp[i][2] = (dp[i - 1][7] + dp[i - 1][9]) % mod;
        dp[i][3] = (dp[i - 1][4] + dp[i - 1][8]) % mod;
        dp[i][4] = (dp[i - 1][0] + dp[i - 1][3] + dp[i - 1][9]) % mod;
        dp[i][6] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][7]) % mod;
        dp[i][7] = (dp[i - 1][2] + dp[i - 1][6]) % mod;
        dp[i][8] = (dp[i - 1][1] + dp[i - 1][3]) % mod;
        dp[i][9] = (dp[i - 1][2] + dp[i - 1][4]) % mod;
        dp[i][0] = (dp[i - 1][4] + dp[i - 1][6]) % mod;
    }
    long long ans = 0;
    for (i = 0; i < 10; i++) {
        ans = (ans + dp[n][i]) % mod;
    }
    return ans;
}


// LC2655
vector<vector<int>> findMaximalUncoveredRanges(int n, vector<vector<int>>& ranges)
{
    vector<vector<int>> combinedRange, ans;
    vector<int> t(2);
    int size = ranges.size();
    int i;

    if (size == 0) {
        return {{0, n - 1}};
    }
    sort(ranges.begin(), ranges.end());
    t[0] = ranges[0][0];
    t[1] = ranges[0][1];
    for (i = 1; i < size; i++) {
        if (ranges[i][0] <= t[1] + 1) {
            t[1] = max(ranges[i][1], t[1]);
        } else {
            combinedRange.emplace_back(t);
            t[0] = ranges[i][0];
            t[1] = ranges[i][1];
        }
    }
    combinedRange.emplace_back(t);

    if (combinedRange[0][0] >= 1) {
        ans.push_back({0, combinedRange[0][0] - 1});
    }
    size = combinedRange.size();
    for (i = 1; i < size; i++) {
        ans.push_back({combinedRange[i - 1][1] + 1, combinedRange[i][0] - 1});
    }
    if (combinedRange[size - 1][1] + 1 < n) {
        ans.push_back({combinedRange[size - 1][1] + 1, n - 1});
    }
    return ans;
}


// LC1334
// 迪杰斯特拉 dijstra
int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold)
{
    int i, k;
    int size = edges.size();
    int cnt;
    vector<int> dist;
    vector<int> cityCnt(n, 0);
    vector<vector<pair<int, int>>> edgeWithWeight(n);
    queue<pair<int, int>> q;
    pair<int, int> t;

    for (i = 0; i < size; i++) {
        edgeWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
        edgeWithWeight[edges[i][1]].push_back({edges[i][0], edges[i][2]});
    }

    for (i = 0; i < n; i++) {
        dist.assign(n, INT_MAX);

        q.push({i, 0});
        while (q.size()) {
            t = q.front();
            q.pop();

            if (dist[t.first] < t.second) {
                continue;
            }
            dist[t.first] = t.second;
            size = edgeWithWeight[t.first].size();
            for (auto e : edgeWithWeight[t.first]) {
                if (e.second + t.second < dist[e.first]) {
                    dist[e.first] = e.second + t.second;
                    q.push({e.first, dist[e.first]});
                }
            }
        }
        cnt = 0;
        for (k = 0; k < n; k++) {
            if (k != i && dist[k] <= distanceThreshold) {
                cnt++;
            }
        }
        cityCnt[i] = cnt;
    }
    int ans = 0;
    int cur = INT_MAX;
    for (i = 0; i < n; i++) {
        if (cityCnt[i] <= cur) {
            cur = cityCnt[i];
            ans = i;
        }
    }
    return ans;
}


// LC2658
void DFSScanGridForFish(vector<vector<int>>& grid, int row, int col, vector<vector<bool>>& visited, int& cnt)
{
    visited[row][col] = true;
    cnt += grid[row][col];

    int i;
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int newR, newC;
    int m = grid.size();
    int n = grid[0].size();
    for (i = 0; i < 4; i++) {
        newR = row + directions[i][0];
        newC = col + directions[i][1];
        if (newR >= 0 && newR < m && newC >= 0 && newC < n && grid[newR][newC] > 0 && visited[newR][newC] == false) {
            DFSScanGridForFish(grid, newR, newC, visited, cnt);
        }
    }
}
int findMaxFish(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int ans, cnt;
    vector<vector<bool>> visited(m, vector<bool>(n, false));

    ans = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (visited[i][j] == false && grid[i][j] > 0) {
                cnt = 0;
                DFSScanGridForFish(grid, i, j, visited, cnt);
                ans = max(ans, cnt);
            }
        }
    }
    return ans;
}


// LC2662
int minimumCost(vector<int>& start, vector<int>& target, vector<vector<int>>& specialRoads) // 超时
{
    int i, j;
    int m = specialRoads.size();
    int r, c, nr, nc;
    int cost, point, npoint;
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    //int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    map<pair<int, int>, vector<tuple<int, int, int>>> trueRoads;
    map<pair<int, int>, long long> costData;
    queue<tuple<int, int, long long>> q;

    for (i = 0; i < m; i++) {
        cost = abs(specialRoads[i][0] - specialRoads[i][2]) +
                abs(specialRoads[i][1] - specialRoads[i][3]);
        if (cost > specialRoads[i][4]) {
            trueRoads[{specialRoads[i][0], specialRoads[i][1]}].push_back(
            {specialRoads[i][2], specialRoads[i][3], specialRoads[i][4]});

            //trueRoads[{specialRoads[i][2], specialRoads[i][3]}].push_back(
            //{specialRoads[i][0], specialRoads[i][1], specialRoads[i][4]});
        }
    }
    int ans = INT_MAX;
    q.push({start[0], start[1], 0});
    while (q.size()) {
        auto t = q.front();
        q.pop();
        r = get<0>(t);
        c = get<1>(t);
        point = get<2>(t);
        //printf ("0: %d %d %d\n", r, c, point);
        if (costData.count({r, c}) == 1 && costData[{r, c}] <= point) {
            continue;
        }
        if (r == target[0] && c == target[1]) {
            ans = min(point, ans);
            continue;
        }
        
        if (r == 1 && c == 2 && point == 1) {
            int ndfd = 34;
        }
        costData[{r, c}] = point;
        if (trueRoads.count({r, c}) == 1) {
            for (auto it : trueRoads[{r, c}]) {
                pair<int, int> p = {get<0>(it), get<1>(it)};
                npoint = point + get<2>(it);
                if (costData.count(p) == 0) {
                    //printf ("1: %d %d %d\n", r, c, point);
                    q.push({p.first, p.second, npoint});
                    //printf ("1.1: %d %d %d\n", p.first, p.second, npoint);
                } else {
                    if (costData[p] > npoint) {
                        // costData[p] = npoint;
                        q.push({p.first, p.second, npoint});
                        //printf ("2: %d %d %d\n", p.first, p.second, npoint);
                    }
                }
            }
        }
        for (i = 0; i < 4; i++) {
            nr = r + directions[i][0];
            nc = c + directions[i][1];
            if (nr < start[0] || nr > target[0] || nc < start[1] || nc > target[1]) {
                continue;
            }
            if (costData.count({nr, nc}) == 1) {
                if (point + 1 >= costData[{nr, nc}]) {
                    continue;
                }
            }
            //printf ("3: %d %d %d\n", nr, nc, point + 1);
            q.push({nr, nc, point + 1});
        }
    }
    return ans;
}

// LC2659
long long countOperationsToEmptyArray(vector<int>& nums) // 有问题
{
    int i;
    int n = nums.size();
    int curIdx;
    long long ans = 0;
    pair<int, int> p;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    for (i = 0; i < n; i++) {
        pq.push({nums[i], i});
    }
    curIdx = 0;
    while (pq.size()) {
        p = pq.top();
        pq.pop();

        if (p.second > curIdx) {
            ans += static_cast<long long>(p.second - curIdx);
        }
    }
    return ans;
}


// LCP78
int rampartDefensiveLine(vector<vector<int>>& rampart)
{
    vector<int> diff;
    int i;
    int n = rampart.size();
    int curLeft;
    int left, right, mid;
    bool tooLarge = false;

    left = 1;
    right = 0;
    for (i = 1; i < n; i++) {
        diff.emplace_back(rampart[i][0] - rampart[i - 1][1]);
        right = max(right, rampart[i][0] + rampart[i - 1][1]);
    }
    n = diff.size();
    if (n == 2) {
        return diff[0] + diff[1];
    }
    int t = 0;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curLeft = diff[0];
        tooLarge = false;
        for (i = 1; i < n; i++) {
            if (curLeft >= mid) {
                curLeft = diff[i];
                continue;
            }
            curLeft = curLeft + diff[i] - mid;
            if (curLeft < 0) {
                right = mid - 1;
                tooLarge = true;
                break;
            }
        }
        if (tooLarge) {
            continue;
        }
        left = mid + 1;
    }
    return right;
}



// LCP80
string CalcEvolutionaryRoute(unordered_map<int, unordered_set<int>>& edges, int cur, unordered_map<int, string>& nodeRoute)
{
    if (edges.count(cur) == 0) {
        return "";
    }
    vector<string> routes;
    string route, ans;
    for (auto it : edges[cur]) {
        route.clear();
        if (nodeRoute.count(it) == 0) {
            route += '0';
            route += CalcEvolutionaryRoute(edges, it, nodeRoute);
            route += '1';
            nodeRoute[it] = route;
        }
        // printf ("%d: %s\n", cur, nodeRoute[it].c_str());
        routes.emplace_back(nodeRoute[it]);
    }
    sort(routes.begin(), routes.end());
    for (auto r : routes) {
        ans += r;
    }
    return ans;
}
string evolutionaryRecord(vector<int>& parents)
{
    int i;
    int n = parents.size();
    int root;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, string> nodeRoute;
    string ans;
    // 建图
    for (i = 0; i < n; i++) {
        if (parents[i] != -1) {
            edges[parents[i]].emplace(i);
        } else {
            root = i;
        }
    }
    ans = CalcEvolutionaryRoute(edges, root, nodeRoute);

    int len = ans.size();
    int idx = len;
    for (i = len - 1; i >= 0; i--) {
        if (ans[i] == '0') {
            idx = i;
            break;
        }
    }
    ans = ans.substr(0, idx + 1);
    return ans;
}


// LC2061
int numberOfCleanRooms(vector<vector<int>>& room)
{
    int m = room.size();
    int n = room[0].size();
    int nx, ny;
    int ans;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<int> cur;
    set<vector<int>> visited;

    cur = {0, 0, 0};
    visited.insert({0, 0, 0});
    while (1) {
        room[cur[0]][cur[1]] = 2;
        nx = cur[0] + directions[cur[2]][0];
        ny = cur[1] + directions[cur[2]][1];

        if (nx < 0 || nx >= m || ny < 0 || ny >= n || room[nx][ny] == 1) {
            cur[2] = (cur[2] + 1) % 4;
        } else {
            cur = {nx, ny, cur[2]};
        }
        if (visited.count(cur) == 1) {
            break;
        }
        visited.insert(cur);
    }
    ans = 0;
    for (auto r : room) {
        for (auto c : r) {
            if (c == 2) {
                ans++;
            }
        }
    }
    return ans;
}


// LC2049
int DFSCountNodeSubNode(int cur, unordered_map<int, vector<int>>& edges, unordered_map<int, pair<int, int>>& nodeCnt)
{
    if (edges.count(cur) == 0) {
        return 1;
    }
    int t;
    if (nodeCnt.count(cur) == 0) {
        t = DFSCountNodeSubNode(edges[cur][0], edges, nodeCnt);
        nodeCnt[cur].first = t;
        if (edges[cur].size() == 2) {
            t = DFSCountNodeSubNode(edges[cur][1], edges, nodeCnt);
            nodeCnt[cur].second = t;
        } else {
            nodeCnt[cur].second = 0;
        }
    }
    return nodeCnt[cur].first + nodeCnt[cur].second + 1;
}
int countHighestScoreNodes(vector<int>& parents)
{
    int i;
    int n = parents.size();
    int leftNodes, rightNodes, parentNodes;
    int root;
    map<long long, int, greater<>> multiply;
    unordered_map<int, vector<int>> edges; // node - {letf, right}
    unordered_map<int, pair<int, int>> nodeCnt; // node - {leftNodeCnt, rightNodeCnt}
    for (i = 0; i < n; i++) {
        if (parents[i] == -1) {
            root = i;
        } else {
            edges[parents[i]].push_back(i);
        }
    }
    DFSCountNodeSubNode(root, edges, nodeCnt);
    /* for (auto it : nodeCnt) {
        printf ("%d : %d %d\n", it.first, it.second.first, it.second.second);
    } */
    for (i = 0; i < n; i++) {
        if (i == root) {
            leftNodes = nodeCnt[i].first;
            rightNodes = nodeCnt[i].second == 0 ? 1 : nodeCnt[i].second;
            multiply[static_cast<long long>(leftNodes) * rightNodes]++;
            continue;
        }
        if (edges.count(i) == 0) { // 叶子节点
            multiply[n - 1]++;
            continue;
        }
        parentNodes = n - nodeCnt[i].first - nodeCnt[i].second - 1;
        leftNodes = nodeCnt[i].first;
        rightNodes = nodeCnt[i].second == 0 ? 1 : nodeCnt[i].second;
        multiply[static_cast<long long>(leftNodes) * rightNodes * parentNodes]++;
    }
    return multiply.begin()->second;
}


// LC2036
long long maximumAlternatingSubarraySum(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<vector<long long>> dp(n, vector<long long>(2, 0)); // dp[i][0] 以i结尾且nums[i] 为正
    long long ans;

    dp[0][0] = nums[0];
    dp[0][1] = -0x3f3f3f3f;
    ans = nums[0];

    for (i = 1; i < n; i++) {
        dp[i][0] = dp[i - 1][1] > 0 ? dp[i - 1][1] + nums[i] : nums[i];
        dp[i][1] = dp[i - 1][0] - nums[i];
        ans = max(ans, dp[i][0]);
        ans = max(ans, dp[i][1]);
    }
    return ans;
}


// LC1218
int longestSubsequence(vector<int>& arr, int difference)
{
    int i;
    int n = arr.size();
    unordered_map<int, int> dp; // dp[arr[i]] - 以arr[i]结尾的最大长度
    int ans ;
    unordered_map<int, vector<int>> arrCnt;
   
    dp[arr[0]] = 1;
    for (i = 1; i < n; i++) {
        if (dp.count(arr[i] - difference) == 1) {
            dp[arr[i]] = dp[arr[i] - difference] + 1;
        } else {
            dp[arr[i]] = 1;
        }
    }
    ans = 0;
    for (auto it : dp) {
        ans = max(ans, it.second);
    }
    return ans;
}


// LC1073
vector<int> addNegabinary(vector<int>& arr1, vector<int>& arr2)
{
    int i;
    int idx;
    int m = arr1.size();
    int n = arr2.size();
    bool f = false;
    vector<int> t(1002, 0);

    reverse(arr1.begin(), arr1.end());
    reverse(arr2.begin(), arr2.end());
    idx = 0;
    while (idx < m && idx < n) {
        t[idx] = t[idx] + arr1[idx] + arr2[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        }
        idx++;
    }
    while (idx < m) {
        t[idx] = t[idx] + arr1[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        }
        idx++;
    }
    while (idx < n) {
        t[idx] = t[idx] + arr2[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        } else if (t[idx] == 3) {
            t[idx] = 1;
            t[idx + 1] = -1;
        }
        idx++;
    }
    if (t[idx] == -1) {
         t[idx] = 1;
        t[idx + 1] = 1;
    }
    vector<int> ans;
    for (i = t.size() - 1; i >= 0; i--) {
        if (f == false && t[i] == 1) {
            f = true;
        }
        if (f) {
            ans.emplace_back(t[i]);
        }
    }
    if (f == false) {
        return {0};
    }
    return ans;
}


// LC1049
int lastStoneWeightII(vector<int>& stones)
{
    int i, j;
    int sum;
    int n = stones.size();

    sum = 0;
    for (auto s : stones) {
        sum += s;
    }
    vector<vector<int>> dp(n + 1, vector<int>(sum / 2 + 1, 0)); // dp[i][j] 前i个石子之和不超过j的最大石子之和
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= sum / 2; j++) {
            if (stones[i - 1] > j) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - stones[i - 1]] + stones[i  - 1]);
            }
        }
    }
    return sum - dp[n][sum / 2] * 2;
}


// LC2434
string robotWithString(string s)
{
    int i;
    int n = s.size();
    string ans;
    char t;
    stack<char> st;
    vector<char> f(n);  // f[i] 表示从i  -  n - 1最小字符

    f[n - 1] = s[n  - 1];
    for (i = n - 2; i >= 0; i--) {
        f[i] = min(f[i + 1], s[i]);
    }
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(s[i]);
            continue;
        }
        t = st.top();
        while (t <= f[i]) {
            ans += t;
            st.pop();
            if (st.empty()) {
                break;
            }
            t = st.top();
        }
        st.push(s[i]);
    }
    while (!st.empty()) {
        ans += st.top();
        st.pop();
    }
    return ans;
}


// LC1377
void DFSFindNode(unordered_map<int, unordered_set<int>>& e, int cur, int from, int target, 
    int cnt, int& step, bool& find, bool& IsLeaf, vector<int>& r, vector<int>& route)
{
    if (find) {
        return;
    }
    if (cur == target) {
        step = cnt;
        find = true;
        if (e[cur].size() == 1 && *e[cur].begin() == from) {
            IsLeaf = true;
        }
        // for (auto it : route) cout << it << " ";
        route = r;
        return;
    }
    for (auto it : e[cur]) {
        if (it != from) {
            r.emplace_back(it);
            DFSFindNode(e, it, cur, target, cnt + 1, step, find, IsLeaf, r, route);
            r.pop_back();
        }
    }
}
double frogPosition(int n, vector<vector<int>>& edges, int t, int target)
{
    if (n == 1) {
        return 1.0;
    }

    int i;
    int ans;
    int step, ways;
    bool IsLeaf, find;
    vector<int> r, route;
    unordered_map<int, unordered_set<int>> e;
    for (auto edge : edges) {
        e[edge[0]].emplace(edge[1]);
        e[edge[1]].emplace(edge[0]);
    }
    step = 0;
    ways = 1;
    IsLeaf = false;
    find = false;
    r.emplace_back(1);
    DFSFindNode(e, 1, -1, target, 0, step, find, IsLeaf, r, route);
    if (step > t) {
        return 0.0;
    }
    if (!IsLeaf && step < t) {
        return 0.0;
    }
    ways *= e[1].size();
    for (i = 1; i < route.size(); i++) {
        ways *= e[route[i]].size() - 1;
    }
    return 1.0 / ways;
}


// LC640
string solveEquation(string equation) // 垃圾题
{
    int i, k;
    int n;
    int idx, m, p;
    int constNum, xNum;
    bool findEqual;
    char sign;
    string t;

    if (equation[0] == '-') {
        equation = "0" + equation;
    }
    idx = equation.find('=');
    if (equation[idx + 1] == '-') {
        equation = equation.substr(0, idx + 1) + "0" + equation.substr(idx + 1);
    }
    idx = 0;
    constNum = xNum = 0;
    findEqual = false;
    sign = '+';
    n = equation.size();
    for (i = 0; i < n; i++) {
        if (equation[i] == '+' || equation[i] == '-') {
            t = equation.substr(idx, i - idx);
            m = t.size();
            if (t[m - 1] == 'x') {
                p = atoi(equation.substr(idx, i - idx).c_str());
                p = p == 0 ? 1 : p;
                if (t[0] == '0') {
                    p = 0;
                }
                for (k = 0; k < m - 1; k++) {
                    if (equation[k] != '0') {
                        break;
                    }
                }
                if (k == m) {

                }
                if (sign == '-') {
                    if (findEqual) {
                        xNum += p;
                    } else {
                        xNum -= p;
                    }
                } else {
                    if (findEqual) {
                        xNum -= p;
                    } else {
                        xNum += p;
                    }
                }
            } else {
                p = atoi(equation.substr(idx, i - idx).c_str());
                if (sign == '-') {
                    if (findEqual) {
                        constNum += p;
                    } else {
                        constNum -= p;
                    }
                } else {
                    if (findEqual) {
                        constNum -= p;
                    } else {
                        constNum += p;
                    }
                }
            }
            sign = equation[i];
            idx = i + 1;
        } else if (equation[i] == '=') {
            findEqual = true;
            t = equation.substr(idx, i - idx);
            m = t.size();
            if (t[m - 1] == 'x') {
                p = atoi(equation.substr(idx, i - idx).c_str());
                p = p == 0 ? 1 : p;
                if (t[0] == '0') {
                    p = 0;
                }
                if (sign == '-') {
                    xNum -= p;
                } else {
                    xNum += p;
                }
            } else {
                p = atoi(equation.substr(idx, i - idx).c_str());
                if (sign == '-') {
                    constNum -= p;
                } else {
                    constNum += p;
                }
            }
            // sign = equation[i + 1] == '-' ? '-' : '+';
            sign = '+';
            idx = i + 1;
        }
    }
    
    t = equation.substr(idx, n - idx);
    m = t.size();
    if (t[m - 1] == 'x') {
        p = atoi(equation.substr(idx, i - idx).c_str());
        p = p == 0 ? 1 : p;
        if (t[0] == '0') {
            p = 0;
        }
        if (sign == '-') {
            xNum += p;
        } else {
            xNum -= p;
        }
    } else {
        p = atoi(equation.substr(idx, i - idx).c_str());
        if (sign == '-') {
            constNum += p;
        } else {
            constNum -= p;
        }
    }

    if (xNum == 0) {
        if (constNum != 0) {
            return "No solution";
        } else {
            return "Infinite solutions";
        }
    }

    int x = constNum * - 1 / xNum;
    string ans = "x=" + to_string(x);
    // cout << ans << endl;
    return ans;
}


// LC2052
int minimumCost(string sentence, int k)
{
    vector<string> words = MySplit(sentence, ' ');
    int i, j;
    int t, ans;
    int n = words.size();
    vector<int> dp(n + 1, 0); // dp[i] - 以第i个单词作为行尾的最小cost
    vector<int> wordLen;

    for (auto w : words) {
        wordLen.emplace_back(w.size());
    }
    dp[1] = (k - wordLen[0]) * (k - wordLen[0]);
    for (i = 2; i <= n; i++) {
        dp[i] = (k - wordLen[i - 1]) * (k - wordLen[i - 1]) + dp[i - 1];
        t = wordLen[i - 1];
        for (j = i - 1; j >= 1; j--) {
            t += 1 + wordLen[j - 1];
            if (t <= k) {
                dp[i] = min(dp[i], (k - t) * (k - t) + dp[j - 1]);
            }
        }
    }
    for (auto d : dp) cout << d << " "; cout << endl;
    t = wordLen[n - 1];
    ans = dp[n - 1];
    for (i = n - 2; i >= 0; i--) {
        t += 1 + wordLen[i];
        if (t <= k) {
            ans = min(ans, dp[i]);
        } else {
            break;
        }
    }
    return ans;
}



// 面试题 04.09. 二叉搜索树序列
vector<vector<int>> BSTSequences(TreeNode* root)
{
    if (root == nullptr) {
        return {{}};
    }

    int i, j;
    int n;
    int nodeNum = TreeNodesNum(root);
    vector<vector<int>> ans;
    vector<int> serial(nodeNum);
    pair<vector<TreeNode *>, unordered_set<TreeNode *>> t; // 已加入点 - 访问点
    queue<pair<vector<TreeNode *>, unordered_set<TreeNode *>>> q;

    q.push({{root}, {root}});
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            auto v = t.first;
            auto s = t.second;
            if (v.size() == nodeNum) {
                for (j = 0; j < v.size(); j++) {
                    serial[j] = v[j]->val;
                }
                ans.emplace_back(serial);
                continue;
            }
            for (j = 0; j < v.size(); j++) {
                if (v[j]->left != nullptr && s.count(v[j]->left) == 0) {
                    v.emplace_back(v[j]->left);
                    s.emplace(v[j]->left);
                    q.push({v, s});
                    v.pop_back();
                    s.erase(v[j]->left);
                }
                if (v[j]->right != nullptr && s.count(v[j]->right) == 0) {
                    v.emplace_back(v[j]->right);
                    s.emplace(v[j]->right);
                    q.push({v, s});
                    v.pop_back();
                    s.erase(v[j]->right);
                }
            }
        }
    }
    return ans;
}


// LC135
int candy(vector<int>& ratings)
{
    int i;
    int n = ratings.size();
    int ans;

    if (n == 1) {
        return 1;
    }
    vector<int> t1(n, 1), t2(n, 1);
    for (i = 1; i < n; i++) {
        if (ratings[i] > ratings[i - 1]) {
            t1[i] = t1[i - 1] + 1;
        }
    }
    // for (auto n : t1) cout << n << " "; cout << endl;

    for (i = n - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            t2[i] = t2[i + 1] + 1;
        }
    }
    // for (auto n : t2) cout << n << " "; cout << endl;

    ans = 0;
    for (i = 0; i < n; i++) {
        ans += max(t1[i], t2[i]);
    }
    return ans;
}


// LC722
vector<string> removeComments(vector<string>& source)
{
    int i, j, k;
    int n = source.size();
    int pos1, pos2, pos3;
    int startIdx;
    vector<string> t = source;
    vector<string> vs;
    // 先处理块注释
    for (i = 0; i < n; i++) {
        pos1 = t[i].find("/*");
        if (pos1 != string::npos) {
            pos2 = t[i].find("//");
            if (pos2 != string::npos) {
                if (pos2 < pos1) {
                    t[i] = t[i].substr(0, pos2);
                    continue;
                }
            }
            startIdx = 0;
            for (j = i; j < n; j++) {
                pos3 = t[j].find("*/", startIdx);
                if (pos3 != string::npos) {
                    if (j == i) {
                        if (pos3 < pos1 + 2) {
                            startIdx = pos1 + 2;
                            j--;
                            continue;
                        }
                        if (pos1 + 1 != pos3) {
                            t[i] = t[i].substr(0, pos1) + t[i].substr(pos3 + 2);
                            i--;
                            break;
                        }
                    } else {
                        t[i] = t[i].substr(0, pos1);
                        t[j] = t[j].substr(pos3 + 2);
                        t[i] += t[j];
                        for (k = i + 1; k <= j; k++) {
                            t[k] = "";
                        }
                        i = j - 1;
                        break;
                    }
                } else {
                    startIdx = 0;
                }
            }
        }
    }
    for (auto line : t) {
        if (line.size() > 0) {
            vs.emplace_back(line);
        }
    }
    for (auto v : vs) cout << v << endl;

    t = vs;
    // 处理行注释
    n = t.size();
    for (i = 0; i < n; i++) {
        pos2 = t[i].find("//");
        if (pos2 != string::npos) {
            t[i] = t[i].substr(0, pos2);
        }
    }
    vs.clear();
    for (auto line : t) {
        if (line.size() > 0) {
            vs.emplace_back(line);
        }
    }
    return vs;
}


// LC1240
void SetRectangle(vector<vector<bool>>& visited, int row, int col, int size, bool val)
{
    int i, j;
    for (i = row; i < row + size; i++) {
        for (j = col; j < col + size; j++) {
            visited[i][j] = val;
        }
    }
}
bool CanTile(vector<vector<bool>>& visited, int row, int col, int size)
{
    int i, j, k;
    int n = visited.size();
    int m = visited[0].size();

    if (row + size > n || col + size > m) {
        return false;
    }
    for (i = row; i < row + size; i++) {
        for (j = col; j < col + size; j++) {
            if (visited[i][j]) {
                return false;
            }
        }
    }
    return true;
}
void Tiling(vector<vector<bool>>& visited, int row, int col, int edgeSize, int cnt, int cntAera, int& ans)
{
    int k;
    int n = visited.size();
    int m = visited[0].size();

    if (cntAera == m * n) {
        ans = min(ans, cnt);
        return;
    }
    if (cnt >= ans || row == n) {
        return;
    }
    if (col == m) {
        Tiling(visited, row + 1, 0, k, cnt, cntAera, ans);
        return;
    }
    if (visited[row][col]) {
        Tiling(visited, row, col + 1, k, cnt, cntAera, ans);
        return;
    }
    for (k = edgeSize; k >= 1; k--) {
        if (CanTile(visited, row, col, k)) {
            SetRectangle(visited, row, col, k, true);
            Tiling(visited, row, col + k, edgeSize, cnt + 1, cntAera + k * k, ans);
            SetRectangle(visited, row, col, k, false);
        }
    }
}
int tilingRectangle(int n, int m)
{
    int small = min(n, m);

    if (small == 1) {
        return n * m;
    }

    int ans = INT_MAX;
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    Tiling(visited, 0, 0, small, 0, 0, ans);
    return ans;
}


// LC132
int minCut(string s)
{
    // dp[i] 表示i结尾的字符串分割为回文子串的最小切割次数
    // dp[0] = 0; 
    int i, j;
    int n = s.size();
    vector<vector<bool>> IsPalindrome(n, vector<bool>(n, false)); // IsPalindrome[i][j] 是否是回文串
    vector<int> dp(n, 0x3f3f3f3f);

    for (j = 1; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                IsPalindrome[i][j] = true;
                continue;
            }
            if (s[i] == s[j]) {
                if (i + 1 == j) {
                    IsPalindrome[i][j] = true;
                } else {
                    IsPalindrome[i][j] = IsPalindrome[i + 1][j - 1];
                }
            } else {
                IsPalindrome[i][j] = false;
            }
        }
    }
    dp[0] = 0;
    for (j = 1; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (IsPalindrome[i][j]) {
                if (i != 0) {
                    dp[j] = min(dp[j], dp[i - 1] + 1);
                } else {
                    dp[j] = 0;
                }
            }
        }
    }
    return dp[n - 1];
}


// LC2731
int sumDistance(vector<int>& nums, string s, int d)
{
    int i;
    int n = s.size();
    int mod = 1000000007;
    vector<long long> prefixSum(n, 0);

    for (i = 0; i < n; i++) {
        nums[i] = static_cast<long long>(nums[i]) + (s[i] == 'R' ? d : -d);
    }
    sort (nums.begin(), nums.end());

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = nums[i] + prefixSum[i - 1];
    }
    long long ans = 0;
    for (i = 0; i < n - 1; i++) {
        ans = (ans + ((prefixSum[n - 1] - prefixSum[i] - static_cast<long long>(n - i - 1) * nums[i] + mod) % mod)) % mod;
    }
    return ans;
}


// LC2735
long long minCost(vector<int>& nums, int x)
{
    int i, k;
    int n = nums.size();
    int t;
    vector<int> idx;
    unordered_map<int, int> val;
    long long sum, s1;
    long long ans = LONG_MAX;

    sum = 0;
    // sum 不旋转的cost
    for (i = 0; i < n; i++) {
        idx.emplace_back(i);
        val[i] = nums[i];
        sum += nums[i];
    }
    ans = min(sum, ans);
    k = 0;
    s1 = 0;
    while (k < n) {
        s1 += x;
        // 旋转idx
        t = idx[0];
        for (i = 1; i < n; i++) {
            idx[i - 1] = idx[i];
        }
        idx[n - 1] = t;

        sum = 0;
        for (i = 0; i < n; i++) {
            if (val[idx[i]] > nums[i]) {
                val[idx[i]] = nums[i];
            }
            sum += val[idx[i]];
        }
        ans = min(ans, sum + s1);
        k++;
    }
    return ans;
}


// LC2737
int minimumDistance(int n, vector<vector<int>>& edges, int s, vector<int>& marked)
{
    int ans;
    vector<int> dist(n, 0x3f3f3f3f);
    unordered_map<int, set<pair<int, int>>> e;
    queue<pair<int, int>> q;

    for (auto edge : edges) {
        e[edge[0]].insert({edge[1], edge[2]});
    }
    q.push({s, 0});
    while (q.size()) {
        auto p = q.front();
        q.pop();
        if (dist[p.first] < p.second) {
            continue;
        }
        dist[p.first] = p.second;
        for (auto edge : e[p.first]) {
            if (edge.second + dist[p.first] < dist[edge.first]) {
                dist[edge.first] = edge.second + dist[p.first];
                q.push({edge.first, dist[edge.first]});
            }
        }
    }
    ans = 0x3f3f3f3f;
    for (auto m : marked) {
        ans = min(ans, dist[m]);
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC588
vector<string> FileSystem::ls(string path)
{
    int i, j;
    int n;
    int m;
    vector<string> ans;
    Trie<string> *node = FileSystem::root;

    if (path == "/") {
        for (i = 0; i < node->children.size(); i++) {
            ans.emplace_back(node->children[i]->val);
        }
        sort (ans.begin(), ans.end());
        return ans;
    }
    vector<string> dirs = MySplit(path, '/');
    n = dirs.size();
    if (FileSystem::fileContent.count(path) == 1) {
        return {dirs[n - 1]};
    }
    for (i = 1; i < n; i++) { // 去掉第一个'/'
        m = node->children.size();
        for (j = 0; j < m; j++) {
            if (node->children[j]->val == dirs[i]) {
                node = node->children[j];
                break;
            }
        }
    }
    for (i = 0; i < node->children.size(); i++) {
        ans.emplace_back(node->children[i]->val);
    }
    sort (ans.begin(), ans.end());
    return ans;
}

void FileSystem::mkdir(string path)
{
    int i, j;
    Trie<string> *node = FileSystem::root;
    vector<string> dirs = MySplit(path, '/');
    int n = dirs.size();
    int m;

    for (i = 1; i < n; i++) { // 去掉第一个'/'
        m = node->children.size();
        for (j = 0; j < m; j++) {
            if (node->children[j]->val == dirs[i]) {
                node = node->children[j];
                break;
            }
        }
        if (j == m) {
            Trie<string> *dir = new Trie<string>(dirs[i]);
            node->children.emplace_back(dir);
            node = dir;
        }
    }
}

void FileSystem::addContentToFile(string filePath, string content)
{
    if (FileSystem::fileContent.count(filePath) == 1) {
        FileSystem::fileContent[filePath] += content;
        return;
    }
    FileSystem::mkdir(filePath);
    FileSystem::fileContent[filePath] = content;
}

string FileSystem::readContentFromFile(string filePath)
{
    return FileSystem::fileContent[filePath];
}


// LC1039
// dp[i][j] - 从顶点i到顶点j构成的多边形最小乘积之和
// dp[i][i + 1] = 0, 所求dp[i][n - 1]
int minScoreTriangulation(vector<int>& values)
{
    int i, j, k;
    int n = values.size();
    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));

    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            if (j == i + 1) {
                dp[i][j] = 0;
            } else if (j == i + 2) {
                dp[i][j] = values[i] * values[i + 1] * values[j];
            } else {
                for (k = i + 1; k < j; k++) {
                    if (k == i + 1) {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[k][j]);
                    } else if (k == j - 1) {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[i][k]);
                    } else {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[i][k] + dp[k][j]);
                    }
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC2762
long long continuousSubarrays(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left;
    int small, big;
    long long ans = 0;
    multiset<int> ms;

    left = 0;
    for (i = 0; i < n; i++) {
        ms.insert(nums[i]);
        small = *ms.begin();
        big = *ms.rbegin();

        while (big - small > 2) {
            ms.erase(ms.find(nums[left])); // C++17新特性可用ms.extract(nums[left]);
            left++;
            small = *ms.begin();
            big = *ms.rbegin();
        }
        ans += i - left + 1;
    }
    return ans;
}


// LC42
int trap(vector<int>& height)
{
    int i;
    int cnt, ans;
    int n = height.size();
    vector<int> h;
    stack<pair<int, int>> st;
    bool start;

    start = false;
    for (i = 0; i < n; i++) {
        if (height[i] == 0) {
            continue;
        } else {
            h.insert(h.end(), height.begin() + i, height.end());
            break;
        }
    }
    n = h.size();
    ans = 0;
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push({h[i], i});
            continue;
        }
        auto t = st.top();
        cnt = 0;
        while (h[i] >= t.first) {
            st.pop();
            if (st.empty()) {
                break;
            }
            cnt++;
            t = st.top();
        }
        ans += cnt;
        st.push({h[i], i});
    }
    return ans;
}


// LC2812
int maximumSafenessFactor(vector<vector<int>>& grid)
{
    int i, j;
    int k;
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> safety(m, vector<int>(n, -1));
    if (grid[0][0] == 1 || grid[m - 1][n - 1] == 1) {
        return 0;
    }
    
    unordered_set<pair<int, int>, MyHash<int, int>> thieves;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                thieves.emplace(make_pair(i, j));
            }
        }
    }
    queue<pair<int, int>> q;
    for (auto t : thieves) {
        q.push(t);
    }
    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int nr, nc;
    int curSafety = 0;
    while (q.size()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            auto p = q.front();
            q.pop();
            if (p.first == 5 && p.second == 0) {
                    auto t = 12;
            }
            if (safety[p.first][p.second] == - 1 || safety[p.first][p.second] > curSafety) {
                safety[p.first][p.second] = curSafety;
            } else {
                continue;
            }

            for (j = 0; j < 4; j++) {
                nr = p.first + directions[j][0];
                nc = p.second + directions[j][1];
                if (nr == 5 && nc == 0) {
                    auto t = 12;
                }
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || thieves.count({nr, nc}) == 1 || 
                    (safety[nr][nc] != -1 && safety[nr][nc] <= curSafety)) {
                    continue;
                }
                q.push({nr, nc});
            }
        }
        curSafety++;
    }
    /* for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            cout << safety[{i, j}] << " ";
        }
        cout << endl;
    } */

    bool canAccess;
    int left, right, mid;

    left = 0;
    right = m + n;
    vector<vector<int>> visited(m, vector<int>(n, -1));
    while (left <= right) {
        mid = (right - left) / 2 + left;
        canAccess = false;
        q.push({0, 0});
        while (q.size()) {
            k = q.size();
            for (i = 0; i < k; i++) {
                auto p = q.front();
                q.pop();

                if (safety[p.first][p.second] < mid || visited[p.first][p.second] == mid) {
                    continue;
                }
                visited[p.first][p.second] = mid;
                if (p == make_pair(m - 1, n - 1)) {
                    canAccess = true;
                    ClearQueue(q);
                    break;
                }
                for (j = 0; j < 4; j++) {
                    nr = p.first + directions[j][0];
                    nc = p.second + directions[j][1];

                    if (nr < 0 || nr >= m || nc < 0 || nc >= n || thieves.count({nr, nc}) == 1
                        || visited[nr][nc] == mid) {
                        continue;
                    }
                    q.push({nr, nc});
                }
            }
        }
        if (canAccess) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right < 0 ? 0 : right;
}


// LC2431
int maxTastiness(vector<int>& price, vector<int>& tastiness, int maxAmount, int maxCoupons)
{
    int i, j, k;
    int n = price.size();
    int ans;
    // dp[i][j][k] 0 - i水果用j金额使用k次优惠券得到的最大甜度
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(maxAmount + 1, vector<int>(maxCoupons + 1, 0)));

    // 买0号水果两种情况
    for (j = price[0]; j <= maxAmount; j++) {
        dp[0][j][0] = tastiness[0];
    }
    for (k = 1; k <= maxCoupons; k++) {
        for (j = price[0] / 2; j <= maxAmount; j++) {
            dp[0][j][k] = tastiness[0];
        }
    }
    for (i = 1; i < n; i++) {
        for (j = 0; j <= maxAmount; j++) {
            for (k = 0; k <= maxCoupons; k++) {
                dp[i][j][k] = dp[i - 1][j][k]; // 不买
                if (j - price[i] >= 0) { // 不用优惠券
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - price[i]][k] + tastiness[i]);
                }
                if (j - price[i] / 2 >= 0 && k > 0) { // 使用优惠券
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - price[i] / 2][k - 1] + tastiness[i]);
                }
            }
        }
    }
    return dp[n - 1][maxAmount][maxCoupons];
}


class Probability {
public:
    double a = 0;
    double b = 0;
    unordered_map<int, int> leftNum, rightNum;
    double f(double t)
    {
        double ans = 1.0;
        while (t > 0) {
            ans *= t;
            t--;
        }
        return ans;
    }
    void DFS(vector<int>& balls, int idx, int leftCnt, int leftLen, int rightCnt, int rightLen, double tol)
    {
        int i;
        double l, r;
        if (idx == balls.size()) {
            l = tol;
            for (auto it : leftNum) {
                l /= f(it.second);
            }
            r = tol;
            for (auto it : rightNum) {
                r /= f(it.second);
            }
            if (leftNum.size() == rightNum.size()) {
                a += l * r;
            }
            b += l * r;

            // cout << left << " " << right << " " << t << endl; 
            // cout << l << " " << r << endl;
            return;
        }
        for (i = 0; i <= balls[idx]; i++) {
            if (leftCnt + i <= leftLen && rightCnt + balls[idx] - i <= rightLen) {
                if (i) {
                    leftNum[idx] += i;
                }
                if (balls[idx] - i) {
                    rightNum[idx] += balls[idx] - i;
                }
                DFS(balls, idx + 1, leftCnt + i, leftLen, rightCnt + balls[idx] - i, rightLen, tol);
                leftNum[idx] == i ? leftNum.erase(idx) : leftNum[idx] -= i;
                rightNum[idx] == balls[idx] - i ? rightNum.erase(idx) : rightNum[idx] -= (balls[idx] - i);
            }
        }
    }
    double getProbability(vector<int>& balls)
    {
        int i;
        int sum;

        sum  = 0;
        for (i = 0; i < balls.size(); i++) {
            sum += balls[i];
        }

        int len = sum / 2;
        DFS(balls, 0, 0, len, 0, len, f(len));
        cout << a << " " << b << endl;
        return a / b;
    }
};

// LC2861
int maxNumberOfAlloys(int n, int k, int budget, vector<vector<int>>& composition, vector<int>& stock, vector<int>& cost)
{
    int i, j;
    int left, right, mid;
    int ans = 0;
    long long val;
    for (i = 0; i < k; i++) {
        left = 0;
        right = 2 * 10e8;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            val = 0;
            for (j = 0; j < n; j++) {
                if (composition[i][j] * static_cast<long long>(mid) > stock[j]) {
                    val += static_cast<long long>(cost[j]) * (composition[i][j] * static_cast<long long>(mid) - stock[j]);
                }
            }
            if (val > budget) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        ans = max(ans, right);
    }
    return ans;
}


// LC2866
void Erase(map<int, int, greater<>>& m, int t)
{
    for (auto it = m.begin(); it != m.end();) {
        if (it->first > t) {
            m.erase(it++);
        } else {
            break;
        }
    }
}
long long maximumSumOfHeights(vector<int>& maxHeights)
{
    int i;
    int curMin, curMax;
    int n = maxHeights.size();
    vector<long long> prefix(n, 0), suffix(n, 0);
    map<int, int, greater<>> m;
    prefix[0] = maxHeights[0];
    curMin = curMax = maxHeights[0];
    m[maxHeights[0]] = 0;
    for (i = 1; i < n; i++) {
        if (maxHeights[i] >= curMax) {
            prefix[i] = prefix[i - 1] + maxHeights[i];
            m[maxHeights[i]] = i;
            if (m.count(maxHeights[i]) == 0) {
                m[maxHeights[i]] = i;
            }
            curMax = maxHeights[i];
        } else {
            if (maxHeights[i] <= curMin) {
                prefix[i] = static_cast<long long>(i + 1) * maxHeights[i];
                curMin = maxHeights[i];
                curMax = maxHeights[i];
                m.clear();
            } else {
                auto it = m.lower_bound(maxHeights[i]);
                curMax = maxHeights[i];
                prefix[i] = prefix[it->second] + static_cast<long long>(i - (it->second + 1) + 1) * maxHeights[i];
                Erase(m, curMax);
            }
        }
        m[maxHeights[i]] = i;
    }
    m.clear();
    suffix[n - 1] = maxHeights[n - 1];
    m[maxHeights[n - 1]] = n - 1;
    curMin = curMax = maxHeights[n - 1];
    for (i = n - 2; i >= 0; i--) {
        if (maxHeights[i] >= curMax) {
            suffix[i] = suffix[i + 1] + maxHeights[i];
            if (m.count(maxHeights[i]) == 0) {
                m[maxHeights[i]] = i;
            }
            curMax = maxHeights[i];
        } else {
            if (maxHeights[i] <= curMin) {
                suffix[i] = static_cast<long long>(n - i) * maxHeights[i];
                curMin = maxHeights[i];
                curMax = maxHeights[i];
                m.clear();
            } else {
                auto it = m.lower_bound(maxHeights[i]);
                curMax = maxHeights[i];
                suffix[i] = suffix[it->second] + static_cast<long long>(it->second - i) * maxHeights[i];
                Erase(m, curMax);
            }
        }
        m[maxHeights[i]] = i;
    }
    long long ans = 0;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            ans = max(ans, suffix[i]);
        } else if (i == n - 1) {
            ans = max(ans, prefix[i]);
        } else {
            ans = max(ans, prefix[i] + suffix[i] - maxHeights[i]);
        }
    }
    // for (auto a : prefix) cout << a << " "; cout << endl;
    // for (auto a : suffix) cout << a << " "; cout << endl;
    return ans;
}


// LC8
int myAtoi(string s)
{
    // 前导空格
    auto lambda1 = [](string &s) {
        int cnt = 0;
        for (auto ch : s) {
            if (ch != ' ') {
                break;
            } else {
                cnt++;
            }
        }
        s = s.substr(cnt);
    };
    function<void(string &)> Trim = lambda1;
    Trim(s);

    // 前导0
    auto lambda2 = [](string &s) {
        int cnt = 0;
        for (auto ch : s) {
            if (ch != '0') {
                break;
            } else {
                cnt++;
            }
        }
        s = s.substr(cnt);
        if (cnt == 0) {
            return false;
        }
        return true;
    };
    function<bool(string &)> Trim0 = lambda2;
    bool operTrim0 = Trim0(s);

    if (s.empty() || s[0] == ' ' || isalpha(s[0])) {
        return 0;
    }

    bool isNegetive = false;
    int startIdx = 0;
    int i;
    if (s[0] == '-') {
        if (operTrim0) {
            return 0;
        }
        isNegetive = true;
        startIdx++;
    } else if (s[0] == '+') {
        if (operTrim0) {
            return 0;
        }
        startIdx++;
    }
    string t;
    for (i = startIdx; i < s.size(); i++) {
        if (isdigit(s[i])) {
            t += s[i];
        } else {
            break;
        }
    }
    if (t.empty()) {
        return 0;
    }
    long long ans = 0;
    for (i = 0; i < t.size(); i++) {
        ans = t[i] - '0' + ans * 10;
        if (isNegetive && ans * -1 < INT_MIN) {
            return INT_MIN;
        }
        if (!isNegetive && ans > INT_MAX) {
            return INT_MAX;
        }
    }
    if (isNegetive) {
        ans *= -1;
    }
    return ans;
}


// LC156
TreeNode* upsideDownBinaryTree(TreeNode* root)
{
    if (root == nullptr) {
        return nullptr;
    }
    stack<pair<TreeNode *, TreeNode *>> st;
    // 递归lambda捕获列表要加上自身引用
    function<void(TreeNode *node, TreeNode *parent)> f = [&st, &f](TreeNode *node, TreeNode *parent) {
        if (node == nullptr) {
            return;
        }
        st.push({node, parent});
        f(node->left, node);
    };

    f(root, nullptr);

    bool flag = true;
    TreeNode *ans = nullptr;
    while (!st.empty()) {
        auto p = st.top();
        st.pop();
        
        if (flag) {
            flag = false;
            ans = p.first;
        }
        p.first->right = p.second;
        if (p.second != nullptr) {
            if (p.second->right != nullptr) {      
                p.first->left = p.second->right;
            }
            p.second->left = nullptr;
            p.second->right = nullptr;
        }
    }
    return ans;
}


// LC174
// dp[i][j] - 从{i, j}走到终点的最小健康点数
int calculateMinimumHP(vector<vector<int>>& dungeon)
{
    int i, j;
    auto m = dungeon.size();
    auto n = dungeon[0].size();
    vector<vector<int>> dp(m, vector<int>(n));

    dp[m - 1][n - 1] = dungeon[m - 1][n - 1] <= 0 ? abs(dungeon[m - 1][n - 1]) + 1 : 1;
    for (j = n - 2; j >= 0; j--) {
        dp[m - 1][j] = max(1, dp[m - 1][j + 1] - dungeon[m - 1][j]);
    }
    for (i = m - 2; i >= 0; i--) {
        dp[i][n - 1] = max(1, dp[i + 1][n - 1] - dungeon[i][n - 1]);
    }
    for (i = m - 2; i >= 0; i--) {
        for (j = n - 2; j >= 0; j--) {
            dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]);
        }
    }
    return dp[0][0];
}


// LC410
// 动态规划
int splitArray(vector<int>& nums, int k)
{
    int i, j, p;
    int n = nums.size();
    // dp[i][k] - 以第i位结束分为k段的最大子数组和的最小值
    vector<vector<int>> dp(n, vector<int>(k + 1, INT_MAX));
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    dp[0][1] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
        dp[i][1] = prefixSum[i];
    }
    for (i = 1; i < n; i++) {
        for (j = 2; j <= k; j++) {
            if (j > i + 1) {
                break;
            }
            for (p = 0; p < i; p++) {
                if (j - 1 > p + 1) {
                    continue;
                }
                dp[i][j] = min(dp[i][j], max(dp[p][j - 1], prefixSum[i] - prefixSum[p]));
            }
        }
    }
    return dp[n - 1][k];
}

// 二分
int splitArray_1(vector<int>& nums, int k)
{
    int i;
    int left, right, mid;
    int cnt, idx;
    int n = nums.size();
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    left = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
        left = max(left, nums[i]);
    }
    right = prefixSum[n - 1];
    while (left <= right) {
        mid = (right - left) / 2 + left;
        cnt = 0;
        idx = 0;
        for (i = 0; i < n; i++) {
            if (cnt == 0) {
                if (prefixSum[i] > mid) {
                    i--;
                    idx = i;
                    cnt++;
                }
            } else {
                if (prefixSum[i] - prefixSum[idx] > mid) {
                    i--;
                    idx = i;
                    cnt++;
                }
            }
        }
        if (cnt >= k) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1094
bool carPooling(vector<vector<int>>& trips, int capacity)
{
    int i, k;
    int n = trips.size();

    sort (trips.begin(), trips.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] != b[1]) {
            return a[1] < b[1];
        }
        return a[2] < b[2]; });
    for (k = 0; k <= 1000; k++) {
        for (i = 0; i < n; i++) {
            if (trips[i][2] == k) {
                capacity += trips[i][0];
            }
            if (trips[i][1] == k) {
                capacity -= trips[i][0];
                if (capacity < 0) {
                    return false;
                }
            }
        }
    }
    return true;
}
// 差分数组
bool carPooling_1(vector<vector<int>>& trips, int capacity)
{
    int i;
    int n = trips.size();
    vector<int> diff(1001, 0);
    // diff[n] = a[n] - a[n - 1];
    for (i = 0; i < n; i++) {
        diff[trips[i][1]] += trips[i][0];
        diff[trips[i][2]] -= trips[i][0];
    }
    vector<int> num(1000, 0);
    num[0] = diff[0];
    if (num[0] > capacity) {
        return false;
    }
    for (i = 1; i < 1000; i++) {
        num[i] = diff[i] + num[i - 1];
        if (num[i] > capacity) {
            return false;
        }
    }
    return true;
}


// LC1488
vector<int> avoidFlood(vector<int>& rains)
{
    int i;
    int n = rains.size();
    unordered_map<int, int> lake; // 第n个湖泊下雨的日期
    set<int> st; // 可抽水的日期
    vector<int> ans(n, 1);
    for (i = 0; i < n; i++) {
        if (rains[i] == 0) {
            st.emplace(i);
        } else {
            ans[i] = -1;
            if (lake.count(rains[i]) == 0) {
                lake[rains[i]] = i;
            } else {
                auto it = st.lower_bound(lake[rains[i]]);
                if (it == st.end()) {
                    ans.clear();
                    return ans;
                }
                ans[*it] = rains[i];
                st.erase(*it);
                lake[rains[i]] = i;
            }
        }
    }
    return ans;
}


// LC188
// dp[i][0 - 1][1 - k] 第i天持有股票已交易k次的最大价值
int maxProfit(int k, vector<int>& prices)
{
    int i, p;
    int n = prices.size();
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(2, vector<int>(k + 1, -0x3f3f3f3f)));

    dp[0][1][1] = -prices[0];
    dp[0][0][0] = 0;
    for (i = 1; i < n; i++) {
        dp[i][0][0] = 0;
        for (p = 1; p <= k; p++) {
            dp[i][0][p] = max(dp[i - 1][0][p], dp[i - 1][1][p] + prices[i]);
            dp[i][1][p] = max(dp[i - 1][1][p], dp[i - 1][0][p - 1] - prices[i]);
        }
    }
    int ans = 0;
    for (i = 0; i < n; i++) {
        for (p = 1; p <= k; p++) {
            // printf ("dp[%d][0][%d] = %d\n", i, p, dp[i][0][p]);
            // printf ("dp[%d][1][%d] = %d\n", i, p, dp[i][1][p]);
            ans = max(ans, dp[i][0][p]);
        }
    }
    return ans;
}


double MySqrt(double num)
{
    double left, right, mid;

    if (num < 0) {
        return INFINITY;
    }
    left = 0.0;
    right = num + 1; // 防止出现num < 1的情况
    while (left <= right) {
        mid = (right - left) / 2 + left;

        auto t = mid * mid;
        if (num - t >= 10e-5) {
            left = mid + 10e-4;
        } else {
            right = mid - 10e-4;
        }
    }
    return right;
}


// LC517
int findMinMoves(vector<int>& machines)
{
    int i;
    int n = machines.size();
    int sum = 0;

    for (auto m : machines) {
        sum += m;
    }
    if (sum % n != 0) {
        return -1;
    }
    int ans = 0;
    int avg = sum / n;
    vector<int> need = machines;
    for (i = 0; i < n; i++) {
        need[i] -= avg;
        ans = max(ans, need[i]);
    }
    for (i = 0; i < n; i++) {
        ans = max(ans, abs(need[i]));
        if (i < n - 1) {
            if (need[i] <= 0) {
                need[i + 1] -= abs(need[i]);
            } else {
                need[i + 1] += need[i];
            }
        }
    }
    return ans;
}


// LC115
int numDistinct(string s, string t)
{
    int i, j;
    int m = s.size();
    int n = t.size();
    int mod = 1000000007;
    if (m < n) {
        return 0;
    }
    // dp[i][j] - s的前i位包含t前j位的子序列个数
    vector<vector<long long>> dp(m + 1, vector<long long>(n + 1, 0));
    for (i = 0; i < m; i++) {
        if (s[i] == t[0]) {
            dp[i + 1][1] = dp[i][1] + 1;
        } else {
            dp[i + 1][1] = dp[i][1];
        }
    }
    for (i = 0; i < m; i++) {
        for (j = 1; j < n; j++) {
            if (i < j) {
                break;
            }
            if (s[i] == t[j]) {
                dp[i + 1][j + 1] = (dp[i][j + 1] + dp[i][j]) % mod;
            } else {
                dp[i + 1][j + 1] = dp[i][j + 1];
            }
        }
    }
    return dp[m][n];
}


// LC1363
string largestMultipleOfThree(vector<int>& digits)
{
    int sum = 0;
    string ans;
    vector<int> remainder1;
    vector<int> remainder2;
    map<int, int, greater<>> data;
    for (auto d : digits) {
        sum += d;
        data[d]++;
        if (d % 3 == 1) {
            remainder1.emplace_back(d);
        } else if (d % 3 == 2) {
            remainder2.emplace_back(d);
        }
    }
    sort (remainder1.begin(), remainder1.end());
    sort (remainder2.begin(), remainder2.end());

    auto func = [&data]() {
        int i;
        string ans;
        for (auto it : data) {
            for (i = 0; i < it.second; i++) {
                ans += it.first + '0';
            }
        }
        return ans;
    };

    if (sum % 3 == 0) {
        ans = func();
    } else if (sum % 3 == 1) { // 去掉一个余1或两个余2
        if (remainder1.size() > 0) {
            auto num = remainder1[0];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }
            ans = func();
        } else if (remainder2.size() > 1) {
            auto num = remainder2[0];
            auto num1 = remainder2[1];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }

            if (data[num1] == 1) {
                data.erase(num1);
            } else {
                data[num1]--;
            }
            ans = func();
        }
    } else { // 去掉一个余2或两个余1
        if (remainder2.size() > 0) {
            auto num = remainder2[0];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }
            ans = func();
        } else if (remainder1.size() > 1) {
            auto num = remainder1[0];
            auto num1 = remainder1[1];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }

            if (data[num1] == 1) {
                data.erase(num1);
            } else {
                data[num1]--;
            }
            ans = func();
        }
    }
    if (ans[0] == '0') {
        return "0";
    }
    return ans;
}


// LC1458
int maxDotProduct(vector<int>& nums1, vector<int>& nums2)
{
    // dp[i][j] - 以nums1前i位 nums2前j位 最大子序列点积
    int i, j;
    int m = nums1.size();
    int n = nums2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, -0x3f3f3f3f));

    dp[0][0] = 0;
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            // 4种情况
            dp[i][j] = max({dp[i][j], nums1[i - 1] * nums2[j - 1] + dp[i - 1][j - 1],
                        nums1[i - 1] * nums2[j - 1],
                        dp[i - 1][j],
                        dp[i][j - 1]});
        }
    }
    return dp[m][n];
}


// LC1298
int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, 
    vector<vector<int>>& containedBoxes, vector<int>& initialBoxes)
{
    int i, k;
    int box;
    int ans = 0;
    unordered_set<int> cantOpen;
    queue<int> q;

    for (auto init : initialBoxes) {
        q.push(init);
    }
    while (q.size()) {
        int size = q.size();
        for (i = 0; i < size; i++) {
            box = q.front();
            q.pop();
            if (status[box] == 1) {
                ans += candies[box];
            } else {
                q.push(box);
                if (cantOpen.count(box) == 1) {
                    goto maxCandiesEND;
                }
                cantOpen.emplace(box);
                continue;
            }
            if (keys[box].size() > 0) {
                for (k = 0; k < keys[box].size(); k++) {
                    status[keys[box][k]] = 1;
                }
            }
            if (containedBoxes[box].size() > 0) {
                for (k = 0; k < containedBoxes[box].size(); k++) {
                    q.push(containedBoxes[box][k]);
                }
            }
        }
    }
maxCandiesEND:
    return ans;
}


// LC1269
int numWays(int steps, int arrLen)
{
    // dp[i][x] - 第i步停留在下标x的方案数, 所求dp[steps][0];
    int i, j;
    int mod = 1000000007;
    vector<vector<long long>> dp(steps + 1, vector<long long>(steps + 2, 0));
    dp[1][0] = 1; // 不动
    if (arrLen > 1) {
        dp[1][1] = 1; // 向右走一步
    }
    for (i = 2; i <= steps; i++) {
        for (j = 0; j < arrLen; j++) {
            if (i < j) {
                break;
            }
            if (j == 0) {
                dp[i][j] = (dp[i - 1][j] + dp[i - 1][j + 1]) % mod;
            } else if (j > 0 && j < arrLen - 1) {
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] + dp[i - 1][j + 1]) % mod;
            } else {
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod;
            }
        }
    }
    return dp[steps][0];
}


// LC2332
int latestTimeCatchTheBus(vector<int>& buses, vector<int>& passengers, int capacity)
{
    int i;
    int left, right, mid;
    int k;
    int bus = buses.size();
    int n = passengers.size();
    unordered_set<int> arriveTime;

    sort(buses.begin(), buses.end());
    sort(passengers.begin(), passengers.end());

    for (auto passenger : passengers) {
        arriveTime.emplace(passenger);
    }

    left = passengers[0];
    right = buses[bus - 1];
    while (left <= right) {
        mid = (right - left) / 2 + left;
        auto busCnt = 0;
        k = 0;
        for (i = 0; i < n; i++) {
            if (passengers[i] <= mid) {
                if (passengers[i] <= buses[busCnt]) {
                    k++;
                    if (k == capacity) {
                        busCnt++;
                        k = 0;
                        if (busCnt == bus) {
                            right = mid - 1;
                            break;
                        }
                    }
                } else {
                    busCnt++;
                    if (busCnt == bus) {
                        right = mid - 1;
                        break;
                    }
                    k = 0;
                    i--;
                    continue;
                }
            } else {
                left = mid + 1;
            }
        }
        if (busCnt < bus) {
            left = mid + 1;
        }
    }
    int ans = right;
    while (arriveTime.count(ans) == 1) {
        ans--;
    }
    return ans;
}


// LC939
int minAreaRect(vector<vector<int>>& points)
{
    int i, j;
    int size = points.size();
    int area = INT_MAX;
    int r = 40000;
    unordered_set<int> dots;

    for (auto p : points) {
        dots.emplace(p[0] * r + p[1]);
    }
    for (i = 0; i < size - 1; i++) {
        for (j = i + 1; j < size; j++) {
            if ((points[i][0] != points[j][0] && points[i][1] != points[j][1]) &&
                (dots.count(points[i][0] * r + points[j][1]) && dots.count(points[j][0] * r + points[i][1]))) {
                area = min(area, abs(points[i][0] - points[j][0]) * abs(points[i][1] - points[j][1]));
            }
        }
    }
    return area == INT_MAX ? 0 : area;
}


// LC593
bool validSquare(vector<int>& p1, vector<int>& p2, vector<int>& p3, vector<int>& p4)
{
    vector<vector<int>> points;
    points.emplace_back(p1);
    points.emplace_back(p2);
    points.emplace_back(p3);
    points.emplace_back(p4);
    double k1, k2;

    vector<vector<int>> tries = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    for (auto tr : tries) {
        auto len1 = (points[tr[0]][0] - points[tr[1]][0]) * (points[tr[0]][0] - points[tr[1]][0]) +
                    (points[tr[0]][1] - points[tr[1]][1]) * (points[tr[0]][1] - points[tr[1]][1]);
        auto len2 = (points[tr[2]][0] - points[tr[3]][0]) * (points[tr[2]][0] - points[tr[3]][0]) +
                    (points[tr[2]][1] - points[tr[3]][1]) * (points[tr[2]][1] - points[tr[3]][1]);
        // 对角线相等
        if (len1 != len2) {
            continue;
        }
        // 中心点一致
        if ((points[tr[0]][0] + points[tr[1]][0] != points[tr[2]][0] + points[tr[3]][0]) || 
            (points[tr[0]][1] + points[tr[1]][1] != points[tr[2]][1] + points[tr[3]][1])) {
            continue;
        }
        // 对角线垂直
        if (points[tr[0]][0] == points[tr[1]][0]) {
            k1 = INT_MAX;
        } else {
            k1 = (points[tr[0]][1] - points[tr[1]][1]) * 1.0 / (points[tr[0]][0] - points[tr[1]][0]);
        }
        if (points[tr[2]][0] == points[tr[3]][0]) {
            k2 = INT_MAX;
        } else {
            k2 = (points[tr[2]][1] - points[tr[3]][1]) * 1.0 / (points[tr[2]][0] - points[tr[3]][0]);
        }
        if ((k1 == 0 && k2 == INT_MAX) || (k1 == INT_MAX && k2 == 0) || (k1 * k2 + 1 < 10e-7)) {
            return true;
        }
    }
    return false;
}


// LC1712
int waysToSplit(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int mod = 1000000007;
    int left, right, mid;
    int idx1, idx2;
    int leftSum, midSum, rightSum;
    long long ans;
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = nums[i] + prefixSum[i - 1];
    }
    ans = 0;
    for (i = 0; i < n - 2; i++) {
        // 找到中间子数组大于左边子数组最小结束下标
        leftSum = prefixSum[i];
        left = i + 1;
        right = n - 2;
        // 所求为left
        while (left <= right)
        {
            mid = (right - left) / 2 + left;
            midSum = prefixSum[mid] - leftSum;
            if (midSum >= leftSum) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        midSum = prefixSum[left] - leftSum;
        rightSum = prefixSum[n - 1] - midSum - leftSum;
        if (rightSum < midSum) {
            continue; // 不能用break
        }
        idx1 = left;

        // 找到中间子数组大于左边子数组最大结束下标
        // leftSum = prefixSum[i];
        left = idx1;
        right = n - 2;
        // 所求为right
        while (left <= right)
        {
            mid = (right - left) / 2 + left;
            midSum = prefixSum[mid] - leftSum;
            rightSum = prefixSum[n - 1] - midSum - leftSum;
            if (midSum <= rightSum) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        idx2 = right;
        ans = (ans + idx2 - idx1 + 1) % mod;
        printf ("i = %d, idx1 = %d, idx2 = %d\n", i, idx1, idx2);
        printf ("%d %d %d\n", prefixSum[i], prefixSum[idx1] - prefixSum[i], prefixSum[n - 1] - prefixSum[idx1]);
    }
    return ans;
}


// LC2398
int maximumRobots(vector<int>& chargeTimes, vector<int>& runningCosts, long long budget)
{
    int i;
    int n = chargeTimes.size();
    int left, right, mid;
    int curTopCharge;
    long long curSum, curCost;
    priority_queue<pair<int, int>, vector<pair<int, int>>> pq;

    left = 1;
    right = n;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curSum = 0;
        for (i = 0; i < mid; i++) {
            pq.push({chargeTimes[i], i});
            curSum += runningCosts[i];
        }
        while (1) {
            auto t = pq.top();
            if (t.second >= mid) {
                pq.pop();
            } else {
                curTopCharge = t.first;
                break;
            }
        }
        curCost = curTopCharge + curSum * mid;
        if (curCost <= budget) {
            left = mid + 1;
            continue;
        }
        for (i = mid; i < n; i++) {
            curSum = curSum - runningCosts[i - mid] + runningCosts[i];
            pq.push({chargeTimes[i], i});
            while (1) {
                auto t = pq.top();
                if (t.second <= i - mid || t.second > i) {
                    pq.pop();
                } else {
                    curTopCharge = t.first;
                    break;
                }
            }
            curCost = curTopCharge + curSum * mid;
            if (curCost <= budget) {
                break;
            }
        }
        if (i == n) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return right;
}


// LC1424
vector<int> findDiagonalOrder(vector<vector<int>>& nums)
{
    int i, j;
    vector<int> ans;
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first + a.second != b.first + b.second) {
            return a.first + a.second > b.first + b.second;
        }
        return a.first < b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);

    for (i = 0; i < nums.size(); i++) {
        for (j = 0; j < nums[i].size(); j++) {
            pq.push({i, j});
        }
    }
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        ans.emplace_back(nums[p.first][p.second]);
    }
    return ans;
}


// LC1191
int kConcatenationMaxSum(vector<int>& arr, int k)
{
    int mod = 1000000007;
    int i;
    int n = arr.size();
    long long curMax, suffixMax;
    vector<long long> prefix(n , 0), suffix(n, 0);

    prefix[0] = arr[0];
    for (i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
    suffix[n - 1] = arr[n - 1];
    suffixMax = arr[n - 1];
    for (i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] + arr[i];
        suffixMax = max(suffixMax, suffix[i]);
    }
    vector<long long> dp(n, 0);
    dp[0] = arr[0];
    curMax = arr[0];
    for (i = 1; i < n; i++) {
        dp[i] = dp[i - 1] > 0 ? dp[i - 1] + arr[i] : arr[i];
        curMax = max(curMax, dp[i]);
    }
    if (k == 1) {
        return curMax < 0 ? 0 : curMax % mod;
    }
    auto t = LONG_LONG_MIN;
    for (i = 0; i < n; i++) {
        if (prefix[n - 1] < 0) {
            t = max({prefix[i] + suffixMax, dp[i], t});
        } else {
            t = max({prefix[i] + suffixMax + (k - 2) * prefix[n - 1], dp[i], t});
        }
    }
    return t < 0 ? 0 : t % mod;
}


// LC1510
bool winnerSquareGame(int n)
{
    // dp[n][0] - 剩余n个石子,alice行动是否必胜
    int i, k;
    vector<vector<bool>> dp(n + 1, vector<bool>(2, false));

    dp[0][0] = false;
    dp[0][1] = true;
    for (i = 1; i <= n; i++) {
        for (k = 1; k <= 320; k++) {
            if (i - k * k >= 0) {
                dp[i][0] = dp[i][0] | dp[i - k * k][1];
                if (dp[i][0]) {
                    break;
                }
            } else {
                break;
            }
        }
        dp[i][1] = !dp[i][0];
    }
    return dp[n][0];
}


// LC1312
int minInsertions(string s)
{
    int i, j;
    int n = s.size();
    // dp[i][j] - 下标i到j的子字符串变成回文串的最小插入字符数
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 0;
            } else if (i + 1 == j) {
                dp[i][j] = (s[i] == s[j] ? 0 : 1);
            } else {
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC1102
int maximumMinimumPath(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int left, right, mid;

    right = left = grid[0][0];
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            left = min(left, grid[i][j]);
            right = max(right, grid[i][j]);
        }
    }
    bool access = false;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        access = false;
        auto t = grid;
        function<void (vector<vector<int>>& grid, int row, int col, bool& access)> DFS = 
            [&mid, &DFS](vector<vector<int>>& grid, int row, int col, bool& access) {
            int i;
            int m = grid.size();
            int n = grid[0].size();
            int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
            if (access) {
                return;
            }
            if (grid[row][col] < mid) {
                return;
            }
            if (row == m - 1 && col == n - 1) {
                access = true;
                return;
            }
            grid[row][col] = -1;
            for (i = 0; i < 4; i++) {
                auto nr = row + directions[i][0];
                auto nc = col + directions[i][1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] == -1 || grid[nr][nc] < mid) {
                    continue;
                }
                DFS(grid, nr, nc, access);
            }
        };
        DFS(t, 0, 0, access);
        if (access) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC1416
int numberOfArrays(string s, int k)
{
    int i, j;
    int n = s.size();
    int mod = 1000000007;
    long long t;
    vector<long long> dp(n + 1, 0);

    if (s[0] - '0' > k) {
        return 0;
    }

    dp[0] = 1;
    dp[1] = 1;
    for (i = 2; i <= n; i++) {
        if (s[i - 1] != '0') {
            dp[i] = dp[i - 1];
        }
        j = i - 2;
        while (j >= 0) {
            if (i - j > 10) {
                break;
            }
            if (s[j] != '0') {
                t = atol(s.substr(j, i - j).c_str());
                if (t > k) {
                    break;
                }
                dp[i] = (dp[i] + dp[j]) % mod;
            }
            j--;
        }
    }
    return dp[n];
}


// LC1301
// BFS超时
vector<int> pathsWithMaxScore_1(vector<string>& board)
{
    int i, j;
    int n = board.size();
    int size;
    int mod = 1000000007;
    int directions[3][2] = {{-1, 0}, {0, -1}, {-1, -1}};
    int points;
    long long ways;
    vector<vector<vector<int>>> boardData(n, vector<vector<int>>(n, vector<int>(2, 0)));
    queue<tuple<int, int, int>> q;

    board[0][0] = '0';
    board[n - 1][n - 1] = '0';
    q.push({n * n - 1, 0, 1});
    while (q.size()) {
        size = q.size();
        for (i = 0; i < size; i++) {
            auto tp = q.front();
            q.pop();

            auto x = get<0>(tp) / n;
            auto y = get<0>(tp) % n;
            points = get<1>(tp);
            ways = get<2>(tp);
            if (points > boardData[x][y][0]) {
                boardData[x][y][0] = points;
                boardData[x][y][1] = ways;
            } else if (points == boardData[x][y][0]) {
                boardData[x][y][1] = (boardData[x][y][1] + ways) % mod;
            } else {
                continue;
            }
            for (j = 0; j < 3; j++) {
                auto nx = x + directions[j][0];
                auto ny = y + directions[j][1];

                if (nx < 0 || ny < 0 || board[nx][ny] == 'X') {
                    continue;
                }
                q.push({nx * n + ny, board[nx][ny] - '0' + points, ways});
            }
        }
    }
    return boardData[0][0];
}

// dp
vector<int> pathsWithMaxScore(vector<string>& board)
{
    int i, j;
    int n = board.size();
    int mod = 1000000007;
    vector<vector<vector<int>>> boardData(n, vector<vector<int>>(n, vector<int>(2, 0)));
    
    // 判断连通性
    bool access = false;
    auto b = board;
    function<void (vector<string>& board, int pos, bool& access)> CanAccess = [&CanAccess](vector<string>& board, int pos, bool& access) {
        if (access) {
            return;
        }
        int i;
        int n = board.size();
        int directions[3][2] = {{-1, 0}, {0, -1}, {-1, -1}};
        int row = pos / n;
        int col = pos % n;

        if (row == 0 && col == 0) {
            access = true;
            return;
        }
        board[row][col] = 'X';
        for (i = 0; i < 3; i++) {
            auto nr = row + directions[i][0];
            auto nc = col + directions[i][1];
            if (nr < 0 || nc < 0 || board[nr][nc] == 'X') {
                continue;
            }
            CanAccess(board, nr * n + nc, access);
        }
    };
    CanAccess(b, n * n - 1, access);
    if (!access) {
        return {0, 0};
    }

    // 预处理
    board[0][0] = '0';
    boardData[n - 1][n - 1] = {0, 1};
    for (i = n - 2; i >= 0; i--) {
        if (board[i][n - 1] != 'X') {
            boardData[i][n - 1] = {board[i][n - 1] - '0' + boardData[i + 1][n - 1][0], 1};
        } else {
            break;
        }
    }
    for (j = n - 2; j >= 0; j--) {
        if (board[n - 1][j] != 'X') {
            boardData[n - 1][j] = {board[n - 1][j] - '0' + boardData[n - 1][j + 1][0], 1};
        } else {
            break;
        }
    }
    int maxPoint;
    vector<vector<int>> waysData;
    for (i = n - 2; i >= 0; i--) {
        for (j = n - 2; j >= 0; j--) {
            if (board[i][j] == 'X') {
                continue;
            }
            waysData.clear();
            if (board[i + 1][j] != 'X' && boardData[i + 1][j][1] > 0) {
                waysData.emplace_back(boardData[i + 1][j]);
            }
            if (board[i][j + 1] != 'X' && boardData[i][j + 1][1] > 0) {
                waysData.emplace_back(boardData[i][j + 1]);
            }
            if (board[i + 1][j + 1] != 'X' && boardData[i + 1][j + 1][1] > 0) {
                waysData.emplace_back(boardData[i + 1][j + 1]);
            }
            // 类似
            // 5X
            // XX
            if (waysData.empty()) {
                continue;
            }
            boardData[i][j][0] = board[i][j] - '0';
            maxPoint = 0;
            for (auto w : waysData) {
                maxPoint = max(maxPoint, w[0]);
            }
            boardData[i][j][0] += maxPoint;
            for (auto w : waysData) {
                if (w[0] == maxPoint) {
                    boardData[i][j][1] = (static_cast<long long>(boardData[i][j][1]) + w[1]) % mod;
                }
            }
        }
    }
    return boardData[0][0];
}


// LC1326
int minTaps(int n, vector<int>& ranges)
{
    int i, j;
    int left, right, prevLeft, prevRight;
    int cnt;
    vector<vector<int>> r;
    for (i = 0; i <= n; i++) {
        if (ranges[i] == 0) {
            continue;
        }
        r.push_back({max(0, i - ranges[i]), min(n, i + ranges[i])});
    }
    if (r.empty()) {
        return -1;
    }
    sort(r.begin(), r.end());

    if (r[0][0] != 0) {
        return -1;
    }
    cnt = 1;
    if (r[0][1] == n) {
        return cnt;
    }
    left = r[0][0];
    right = r[0][1];
    prevRight = -1;
    for (i = 1; i < r.size(); i++) {
        if (r[i][0] == left) {
            right = r[i][1];
        } else if (r[i][0] > right) {
            return -1;
        } else if (r[i][1] > right) {
            if (r[i][0] > prevRight) {
                prevRight = right;
                prevLeft = left;
                right = r[i][1];
                left = r[i][0];
                cnt++;
            } else {
                right = r[i][1];
            }
        }
        if (r[i][1] == n) {
            break;
        }
    }
    return right < n ? - 1 : cnt;
}

// LC1964
// 求最长不降子序列, 二分法
vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles)
{
    int i;
    int n = obstacles.size();
    int m;
    int left, right, mid;
    vector<int> low;
    vector<int> ans(n, 0);
    low.emplace_back(obstacles[0]);
    ans[0] = 1;
    for (i = 1; i < n; i++) {
        m = low.size();
        if (obstacles[i] >= low[m - 1]) {
            low.emplace_back(obstacles[i]);
            ans[i] = low.size();
            continue;
        }
        left = 0;
        right = m - 1;
        while (left <= right) { // 所求为left
            mid = ((right - left) >> 1) + left;
            if (low[mid] <= obstacles[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        low[left] = obstacles[i];
        ans[i] = left + 1;
    }
    return ans;
}


// LC907
// 样例 [71,55,82,55] - 593
int sumSubarrayMins(vector<int>& arr)
{
    // 分别找某一个数左右第一个小于它的值, 两个单调递增栈
    int i, idx;
    int n = arr.size();
    unordered_map<int, pair<int, int>> idxBoundary; // 下标 - {左长度, 右长度}
    stack<int> stRight, stLeft;

    // 右边
    for (i = 0; i < n; i++) {
        if (stRight.empty()) {
            stRight.push(i);
            continue;
        }
        idx = stRight.top();
        if (arr[idx] < arr[i]) {
            stRight.push(i);
        } else {
            while (arr[idx] >= arr[i]) {
                idxBoundary[idx].second = i - idx;
                stRight.pop();
                if (stRight.empty()) {
                    break;
                }
                idx = stRight.top();
            }
            stRight.push(i);
        }
    }
    while (!stRight.empty()) {
        idx = stRight.top();
        idxBoundary[idx].second = n - idx;
        stRight.pop();
    }

    // 左边
    for (i = n - 1; i >= 0; i--) {
        if (stLeft.empty()) {
            stLeft.push(i);
            continue;
        }
        idx = stLeft.top();
        if (arr[i] >= arr[idx]) { // 注意此处判断关系与上面不同,是为了避免有相同元素重复记录子数组
            stLeft.push(i);
        } else {
            while (arr[i] < arr[idx]) {  // 这里也是
                idxBoundary[idx].first = idx - i;
                stLeft.pop();
                if (stLeft.empty()) {
                    break;
                }
                idx = stLeft.top();
            }
            stLeft.push(i);
        }
    }
    while (!stLeft.empty()) {
        idx = stLeft.top();
        idxBoundary[idx].first = idx + 1;
        stLeft.pop();
    }
    long long ans = 0;
    for (auto it : idxBoundary) {
        ans += static_cast<long long>(it.second.first) * it.second.second * arr[it.first];
    }
    return ans % 1000000007;
}


// LC755
vector<int> pourWater(vector<int>& heights, int volume, int k)
{
    int i;
    int width = heights.size();
    int curHeight, curIdx;
    while (volume) {
        curHeight = heights[k] + 1;
        curIdx = k;
        for (i = k - 1; i >= 0; i--) {
            if (heights[i] + 1 < curHeight) {
                curHeight = heights[i] + 1;
                curIdx = i;
            } else if (heights[i] >= curHeight) {
                break;
            }
        }
        if (curHeight == heights[k] + 1) {
            curIdx = k;
        }
        if (curIdx == k) {
            for (i = k + 1; i < width; i++) {
                if (heights[i] + 1 < curHeight) {
                    curHeight = heights[i] + 1;
                    curIdx = i;
                } else if (heights[i] >= curHeight) {
                    break;
                }
            }
        }
        heights[curIdx] = curHeight;
        volume--;
    }
    return heights;
}


// LC828
int uniqueLetterString(string s)
{
    int i;
    int n = s.size();
    vector<vector<int>> alphaIdx(26, vector<int>(2, -1)); // 上一个字符出现位置
    vector<int> f(n, 0); // f[i] - 以s[i]结尾的子字符串的唯一字符的个数
    vector<int> dp(n, 0); // dp[i] - 到s[i]为止所有子字符串唯一字符的个数

    f[0] = 1;
    dp[0] = 1;
    alphaIdx[s[i] - 'A'][0] = 0;
    for (i = 1; i < n; i++) {
        if (alphaIdx[s[i] - 'A'][0] == -1) {
            f[i] = f[i - 1] + i + 1;
            alphaIdx[s[i] - 'A'][0] = i;
        } else {
            if (alphaIdx[s[i] - 'A'][1] == -1) {
                f[i] = f[i - 1] + i + 1 - 2 * (alphaIdx[s[i] - 'A'][0] + 1);
                alphaIdx[s[i] - 'A'][1] = alphaIdx[s[i] - 'A'][0];
                alphaIdx[s[i] - 'A'][0] = i;
            } else {
                auto len1 = alphaIdx[s[i] - 'A'][0] + 1;
                auto len2 = alphaIdx[s[i] - 'A'][1] + 1;
                f[i] = f[i - 1] + i + 1 - 2 * (len1 - len2) - len2;
                alphaIdx[s[i] - 'A'][1] = alphaIdx[s[i] - 'A'][0];
                alphaIdx[s[i] - 'A'][0] = i;
            }
        }
        dp[i] = dp[i - 1] + f[i];
    }
    return dp[n - 1];
}


// LC1411
int numOfWays(int n)
{
    int i;
    int mod = 1000000007;
    // 对于每一行, 可以有12种方式, 其中2色6种, 3色6种;
    // 2色可衍生出5种配对, 3色2种, 2色3种; 3色可衍生4种, 各2种
    // dp[i][0] - 第i层以双色结尾的方案数
    vector<vector<long long>> dp(n, vector<long long>(2, 0));

    dp[0][0] = 6;
    dp[0][1] = 6;

    for (i = 1; i < n; i++) {
        dp[i][0] = (dp[i - 1][0] * 3 + dp[i - 1][1] * 2) % mod;
        dp[i][1] = (dp[i - 1][0] * 2 + dp[i - 1][1] * 2) % mod;
    }
    return (dp[n - 1][0] + dp[n - 1][1]) % mod;
}


// LC790
int numTilings(int n)
{
    int i;
    int mod = 1000000007;
    vector<long long> dp(n + 1, 0);

    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return 2;
    }
    // 长为n的地板可分为n-i和i两部分, 其中i具有唯一性, 比如i=2的情况只能是上下两横向,两竖向则与i=1重复计数了
    // f[n] = f[n - 1] + f[n - 2] + 2 * f[n - 3] + 2 * f[n - 4] + ... + 2 * f[0]
    // f[0] = 1;
    dp[0] = 1;
    dp[1] = 1;
    dp[2] = 2;
    long long prefixSum = dp[0];
    for (i = 3; i <= n; i++) {
        dp[i] = (dp[i - 1] + dp[i - 2] + prefixSum * 2) % mod;
        prefixSum = (prefixSum + dp[i - 2]) % mod;
    }
    return dp[n];
}


// LC1406
string stoneGameIII(vector<int>& stoneValue)
{
    int i;
    int a, b, c;
    int n = stoneValue.size();
    // dp[i] - alice从第i处的石堆开始, 能获得的最大和
    vector<int> dp(n, INT_MIN);
    // 后缀和, 方便计算
    vector<int> suffixSum(n, 0);
    suffixSum[n - 1] = stoneValue[n - 1];
    for (i = n - 2; i >= 0; i--) {
        suffixSum[i] = suffixSum[i + 1] + stoneValue[i];
    }

    dp[n - 1] = stoneValue[n - 1];
    for (i = n - 2; i >= 0; i--) {
        a = INT_MIN;
        if (i + 1 < n) {
            a = suffixSum[i] - dp[i + 1];
        }
        b = INT_MIN;
        if (i + 2 < n) {
            b = suffixSum[i] - dp[i + 2];
        }
        c = INT_MIN;
        if (i + 3 < n) {
            c = suffixSum[i] - dp[i + 3];
        }
        dp[i] = max({dp[i], a, b, c});
        if (i + 3 >= n) { // 直接取完
            dp[i] = max(dp[i], suffixSum[i]);
        } 
    }
    int alice = dp[0];
    int bob = suffixSum[0] - alice;

    if (alice > bob) {
        return "Alice";
    } else if (alice < bob) {
        return "Bob";
    }
    return "Tie";
}


// LC1686
int stoneGameVI(vector<int>& aliceValues, vector<int>& bobValues)
{
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first + a.second < b.first + b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);

    int i;
    int n = aliceValues.size();
    for (i = 0; i < n; i++) {
        pq.push({aliceValues[i], bobValues[i]});
    }
    int cnt;
    int alice, bob;

    cnt = alice = bob = 0;
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        if (cnt % 2 == 0) {
            alice += p.first;
        } else {
            bob += p.second;
        }
        cnt++;
    }
    if (alice > bob) {
        return 1;
    } else if (alice < bob) {
        return -1;
    }
    return 0;
}


// LC1690
int stoneGameVII(vector<int>& stones)
{
    int i, j;
    int head, tail;
    int n = stones.size();
    vector<vector<int>> dpMax(n, vector<int>(n, 0)); // dpMax[i][j] alice先手在区间[i,j]得到的最大分差

    vector<int> prefixSum(n, 0);
    prefixSum[0] = stones[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + stones[i];
    }
    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dpMax[i][j] = 0;
                continue;
            } else if (i + 1 == j) {
                dpMax[i][j] = max(stones[i], stones[j]);
                continue;
            }
            if (i == 0) {
                head = prefixSum[j - 1]; // i -> j - 1
            } else {
                head = prefixSum[j - 1] - prefixSum[i - 1];
            }
            tail = prefixSum[j] - prefixSum[i]; // i + 1 -> j
            dpMax[i][j] = max(head - dpMax[i][j - 1], tail - dpMax[i + 1][j]); // 此处减去dpMax为bob的最佳策略
        }
    }
    return dpMax[0][n - 1];
}


// LC1562
// 从后往前二分
int findLatestStep(vector<int>& arr, int m)
{
    int i;
    int n = arr.size();
    set<int> blankInc;
    set<int, greater<>> blankDec;
    if (m == n) {
        return n;
    }
    for (i = 0; i < n; i++) {
        arr[i]--;
    }
    blankInc.emplace(arr[n - 1]);
    blankDec.emplace(arr[n - 1]);
    if (arr[n - 1] == m || n - 1 - arr[n - 1] == m) {
        return n - 1;
    }
    int len;
    for (i = n - 2; i >= 0; i--) {
        // 找最近的左右端点
        // 右端点
        auto it = blankInc.lower_bound(arr[i]);
        if (it == blankInc.end()) {
            len = n - 1 - arr[i];
        } else {
            len = *it - 1 - arr[i];
        }
        if (len == m) {
            return i;
        }
        // 左端点
        it = blankDec.lower_bound(arr[i]);
        if (it == blankDec.end()) {
            len = arr[i];
        } else {
            len = arr[i] - 1 - *it;
        }
        if (len == m) {
            return i;
        }
        blankInc.emplace(arr[i]);
        blankDec.emplace(arr[i]);
    }
    return -1;
}


// LC774
double minmaxGasDist(vector<int>& stations, int k)
{
    int i;
    int n = stations.size();
    long long curPos, cnt, t;
    long long left, right, mid;
    vector<long long> st(n);

    for (i = 0; i < n; i++) {
        st[i] = stations[i] * 1e5;
    }

    left = 1;
    right = 1e8 * 1e5;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curPos = st[0];
        t = k;
        for (i = 1; i < n; i++) {
            if (st[i] - curPos > mid) {
                if ((st[i] - curPos) % mid == 0) {
                    cnt = (st[i] - curPos) / mid - 1;
                } else {
                    cnt = (st[i] - curPos) / mid;
                }
                curPos = st[i];
                t -= cnt;
                if (t < 0) {
                    left = mid + 1;
                    break;
                }
            } else {
                curPos = st[i];
            }
        }
        if (i == n) {
            right = mid - 1;
        }
    }
    return left / 1e5;
}


// LC776
// 如果当前节点 root->val <= target, 那它及左子树都应该被划分到less树, 但root的右子树中也有可能存在小于等于target的节点,
// 而这些节点可以通过 p = splitBST(root->right, target) 返回出来, 而由于是右子树分出来的, 子树p[0]中所有结点的值肯定也比root->val大,
// 因此接在root的右边 root->right = p[0];
// root->val > target时同理, root及其右子树直接划分出来到more, 递归左子树找漏掉的大值节点
vector<TreeNode*> splitBST(TreeNode* root, int target)
{
    if (root == nullptr) {
        return {nullptr, nullptr};
    }
    TreeNode *less, *greater;
    if (root->val <= target) {
        less = root;
        auto p = splitBST(root->right, target);
        root->right = p[0];
        greater = p[1];
    } else {
        greater = root;
        auto p = splitBST(root->left, target);
        root->left = p[1];
        less = p[0];
    }
    return {less, greater};
}


// LC801
int minSwap(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int n = nums1.size();
    int a1, a2, b1, b2;
    vector<vector<int>> dp(n, vector<int>(2, 0x3f3f3f3f));

    // dp[n][0] - 第n位不交换使序列严格递增的最小交换次数
    dp[0][0] = 0;
    a1 = nums1[0];
    b1 = nums2[0];
    dp[0][1] = 1;
    a2 = nums2[0];
    b2 = nums1[0];

    for (i = 1; i < n; i++) {
        if (nums1[i] > a1 && nums2[i] > b1) {
            dp[i][0] = dp[i - 1][0];
        }
        if (nums1[i] > a2 && nums2[i] > b2) {
            dp[i][0] = min(dp[i - 1][1], dp[i][0]);
        }

        if (nums2[i] > a1 && nums1[i] > b1) {
            dp[i][1] = dp[i - 1][0] + 1;
        }
        if (nums2[i] > a2 && nums1[i] > b2) {
            dp[i][1] = min(dp[i - 1][1] + 1, dp[i][1]);
        }
        a1 = nums1[i];
        b1 = nums2[i];
        a2 = nums2[i];
        b2 = nums1[i];
    }
    return min(dp[n - 1][0], dp[n - 1][1]);
}


// LC1621
int numberOfSets(int n, int k)
{
    int mod = 1000000007;
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, 0));
    vector<vector<long long>> sum(n + 1, vector<long long>(k + 1, 0));

    // dp[n][1] = (n - 1) + (n - 2) + ... + 1 = n(n - 1) / 2 (n >= 2)
    // dp[n][k] = dp[n - 1][k - 1] + dp[n - 2][k - 1] + ... + dp[k][k - 1] + (dp[n - 2][k - 1] + ... + dp[k][k - 1]) + ...   (k > 1)
    // 例如
    // dp[5][2] = (dp[4][1] + dp[3][1] + dp[2][1]) + (dp[3][1] + dp[2][1]) + (dp[2][1]) = 15
    // dp[4][2] = (dp[3][1] + dp[2][1]) + (dp[2][1]) = 5
    // dp[5][2] = dp[4][2] + sum(dp[5 - 1][2 - 1], dp[5 - 2][2 - 1], dp[5 - 3][2 - 1])
    // 再例如 dp[6][3] = (dp[5][2] + dp[4][2] + dp[3][2]) + (dp[4][2] + dp[3][2]) + (dp[3][2]) = dp[5][3] +
    // sum(dp[5][2], dp[4][2], dp[3][2])
    int i, j;
    for (i = 2; i <= n; i++) {
        for (j = 1; j <= k; j++) {
            if (j == 1) {
                dp[i][j] = (i - 1) * i / 2 % mod;
                if (i == 2) {
                    sum[i][j] = dp[i][j];
                } else {
                    sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
                }
                continue;
            } else if (i == j + 1) {
                dp[i][j] = 1;
                if (i == 2) {
                    sum[i][j] = dp[i][j];
                } else {
                    sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
                }
                continue;
            }
            if (j >= i) {
                break;
            }
           dp[i][j] = (sum[i - 1][j - 1] + dp[i - 1][j]) % mod;
           sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
        }
    }
    return dp[n][k];
}


// LC2477
long long minimumFuelCost(vector<vector<int>>& roads, int seats)
{
    if (roads.empty()) {
        return 0;
    }
    int i;
    int size = roads.size();
    int parent;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, int> nodeData; // 节点 - 人数
    for (i = 0; i <= size; i++) {
        nodeData[i] = 1;
    }
    for (auto r : roads) {
        edges[r[0]].emplace(r[1]);
        edges[r[1]].emplace(r[0]);
    }
    queue<pair<int, int>> q;
    for (auto it : edges) {
        // 叶子节点
        if (it.second.size() == 1) {
            q.push({it.first, -1});
        }
    }

    while (q.size()) {
        auto p = q.front();
        q.pop();
        if (p.first == 0) {
            continue;
        }
        parent = *edges[p.first].begin();
        edges[parent].erase(p.first);
        nodeData[parent] += nodeData[p.first];
        if (edges[parent].size() == 1) {
            q.push({parent, p.first});
        }
    }
    long long ans = 0;
    for (auto it : nodeData) {
        // printf ("%d %d\n", it.first, it.second);
        if (it.first == 0 || it.first == -1) {
            continue;
        }
        if (it.second % seats == 0) {
            ans += it.second / seats;
        } else {
            ans += it.second / seats + 1;
        }
    }
    return ans;
}


// LC2955
vector<int> sameEndSubstringCount(string s, vector<vector<int>>& queries)
{
    // unordered_map<char, vector<int>> idx; 超时
    vector<vector<int>> idx(26);
    int i;
    int n = s.size();
    int left, right, mid;
    int leftBound, rightBound, sum;
    vector<int> ans;
    for (i = 0; i < n; i++) {
        idx[s[i] - 'a'].emplace_back(i);
    }

    for (auto q : queries) {
        sum = 0;
        for (auto ch = 'a'; ch <= 'z'; ch++) {
            if (idx[ch - 'a'].size() == 0) {
                continue;
            }
            // 第一个大于等于q[0]的idx[ch], left
            left = 0;
            right = idx[ch - 'a'].size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (idx[ch - 'a'][mid] < q[0]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            if (left >= idx[ch - 'a'].size()) {
                continue;
            }
            leftBound = left;

            // 第一个小于等于q[1]的idx[ch], right
            left = 0;
            right = idx[ch - 'a'].size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (idx[ch - 'a'][mid] > q[1]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            if (right < 0) {
                continue;
            }
            rightBound = right;
            auto t = (rightBound - leftBound + 1) * (rightBound - leftBound + 2) / 2;
            sum += t;
        }
        ans.emplace_back(sum);
    }
    return ans;
}


// LC2267
bool hasValidPath(vector<vector<char>>& grid)
{
    int m = grid.size();
    int n = grid[0].size();
    unordered_map<int, unordered_set<int>> route;
    if (grid[0][0] == ')' || grid[m - 1][n - 1] == '(') {
        return false;
    }
    if ((m + n - 1) % 2 == 1) { // 左右括号数量要匹配
        return false;
    }
    function<void (vector<vector<char>>&, int, int, int, int, bool&)> DFS =
    [&DFS, &route](vector<vector<char>>& grid, int row, int col, int cnt1, int cnt2, bool& find) {
        if (find) {
            return;
        }

        int k;
        int m = grid.size();
        int n = grid[0].size();
        int directions[2][2] = {{0, 1}, {1, 0}};

        if (row == m - 1 && col == n - 1 && cnt1 == cnt2) {
            find = true;
            return;
        }
        if (route.count(row * n + col) && route[row * n + col].count(cnt1 - cnt2)) {
            return;
        }
        route[row * n + col].emplace(cnt1 - cnt2);
        auto t = m - 1 - row + n - 1 - col;
        if (cnt1 < cnt2 || cnt1 * 2 > m + n || (cnt1 == cnt2 && t % 2 != 0) || (cnt1 > cnt2 && t < cnt1 - cnt2)) {
            return;
        }

        for (k = 0; k < 2; k++) {
            auto nr = row + directions[k][0];
            auto nc = col + directions[k][1];
            if (nr < m && nc < n) {
                if (grid[nr][nc] == '(') {
                    DFS(grid, nr, nc, cnt1 + 1, cnt2, find);
                } else {
                    DFS(grid, nr, nc, cnt1, cnt2 + 1, find);
                }
            }
        }
    };
    bool find = false;
    DFS(grid, 0, 0, 1, 0, find);
    return find;
}


// LC2763
int sumImbalanceNumbers(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int ans, cnt, curMaxVal, curMinVal;
    map<int, int> dataLess;
    map<int, int, greater<>> dataGreater;

    ans = 0;
    for (i = 0; i < n; i++) {
        dataLess.clear();
        dataGreater.clear();
        cnt = 0;
        for (j = i; j < n; j++) {
            if (j == i) {
                dataLess[nums[j]]++;
                dataGreater[nums[j]]++;
                continue;
            }
            if (dataLess.count(nums[j])) {
                dataLess[nums[j]]++;
                dataGreater[nums[j]]++;
                ans += cnt;
                continue;
            }
            if (nums[j] < dataLess.begin()->first) { // 最小值
                curMinVal = dataLess.begin()->first;
                if (curMinVal - nums[j] > 1) {
                    cnt++;
                }
            } else if (nums[j] > dataGreater.begin()->first) { // 最大值
                curMaxVal = dataGreater.begin()->first;
                if (nums[j] - curMaxVal > 1) {
                    cnt++;
                }
            } else {
                auto rightIter = dataLess.upper_bound(nums[j]);
                auto leftIter = dataGreater.upper_bound(nums[j]);
                if (nums[j] - leftIter->first > 1) {
                    if (rightIter->first - nums[j] > 1) {
                        cnt++;
                    } else {
                        // cnt不变
                    }
                } else {
                    if (rightIter->first - nums[j] > 1) {
                        // cnt不变
                    } else {
                        cnt--;
                    }
                }
            }
            dataLess[nums[j]]++;
            dataGreater[nums[j]]++;
            ans += cnt;
        }
    }
    return ans;
}


// LC2008
long long maxTaxiEarnings(int n, vector<vector<int>>& rides)
{
    int i, j;
    vector<long long> dp(n + 1, 0);

    sort(rides.begin(), rides.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] != b[1]) {
            return a[1] < b[1];
        }
        return a[0] < b[0];
    });
    j = 0;
    for (i = 2; i <= n; i++) {
        if (j == rides.size() || i < rides[j][1]) {
            dp[i] = max(dp[i], dp[i - 1]);
        } else {
            dp[i] = max(dp[i], dp[rides[j][0]] + rides[j][1] - rides[j][0] + rides[j][2]);
            i--;
            j++;
        }
    }
    return dp[n];
}


// LC1671
int minimumMountainRemovals(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    vector<int> dpUp(n, 1), dpDown(n, 1);

    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) {
                dpUp[i] = max(dpUp[i], dpUp[j] + 1);
            }
        }
    }
    for (i = n - 2; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            if (nums[i] > nums[j]) {
                dpDown[i] = max(dpDown[i], dpDown[j] + 1);
            }
        }
    }
    int ans = INT_MAX;
    for (i = 1; i < n - 1; i++) {
        if (dpUp[i] != 1 && dpDown[i] != 1) {
            ans = min(ans, n - (dpUp[i] + dpDown[i] - 1));
        }
    }
    return ans;
}


// LC2959
// 迪杰斯特拉
int numberOfSets(int n, int maxDistance, vector<vector<int>>& roads)
{
    // 一共有 2^n 种情况
    int i, j, k;
    int tmp, idx;
    int ans = 0;
    vector<int> status(n);
    vector<int> dist(n);
    vector<vector<pair<int, int>>> edgeWithWeight(n);
    queue<pair<int, int>> q;
    pair<int, int> t;
    for (k = 0; k < pow(2, n); k++) {
        tmp = k;
        idx = 0;
        while (tmp) {
            status[idx] = tmp % 2;
            tmp /= 2;
            idx++;
        }
        for (i = 0; i < n; i++) {
            edgeWithWeight[i].clear();
        }
        for (auto r : roads) {
            if (status[r[0]] && status[r[1]]) { // 当前情况存在路径
                edgeWithWeight[r[0]].push_back({r[1], r[2]});
                edgeWithWeight[r[1]].push_back({r[0], r[2]});
            }
        }
        for (i = 0; i < n; i++) {
            if (status[i] == 0) {
                continue;
            }
            dist.assign(n, INT_MAX);
            q.push({i, 0});
            while (q.size()) {
                t = q.front();
                q.pop();

                if (dist[t.first] < t.second) {
                    continue;
                }
                dist[t.first] = t.second;
                auto size = edgeWithWeight[t.first].size();
                for (auto e : edgeWithWeight[t.first]) {
                    if (e.second + t.second < dist[e.first]) {
                        dist[e.first] = e.second + t.second;
                        q.push({e.first, dist[e.first]});
                    }
                }
            }
            for (j = 0; j < n; j++) {
                if (j != i && status[j] && dist[j] > maxDistance) {
                    goto numberOfSetsENDLOOP;
                }
            }
        }
        ans++;
numberOfSetsENDLOOP:
        ;
    }
    return ans;
}


// LC1631
int minimumEffortPath(vector<vector<int>>& heights)
{
    int i, j;
    int left, right, mid;
    int m = heights.size();
    int n = heights[0].size();
    vector<vector<bool>> visited(heights.size(), vector<bool>(heights[0].size(), false));
    bool reach = false;

    left = right = 0;
    for (auto height : heights) {
        for (auto h : height) {
            right = max(right, h);
        }
    }
    while (left <= right) {
        mid = (right - left) / 2 + left;
        function<void (int, int, vector<vector<bool>>&, bool&)> DFS = 
            [&DFS, &heights, &mid](int row, int col, vector<vector<bool>>& visited, bool& reach) -> void {

            int k;
            int m = heights.size();
            int n = heights[0].size();
            int direction[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            if (reach) {
                return;
            }
            if (row == m - 1 && col == n - 1) {
                reach = true;
                return;
            }
            visited[row][col] = true;
            for (k = 0; k < 4; k++) {
                auto nr = row + direction[k][0];
                auto nc = col + direction[k][1];

                if (nr < 0 || nr >= m || nc < 0 || nc >= n || 
                    visited[nr][nc] || abs(heights[row][col] - heights[nr][nc]) > mid) {
                    continue;
                }
                DFS(nr, nc, visited, reach);
            }
        };
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                visited[i][j] = false;
            }
        }
        reach = false;
        DFS(0, 0, visited, reach);
        if (reach) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC1818
int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int n = nums1.size();
    long long sum;
    set<int> less;
    set<int, greater<>> greater;

    sum = 0;
    for (i = 0; i < n; i++) {
        sum += abs(nums1[i] - nums2[i]);
        less.emplace(nums1[i]);
        greater.emplace(nums1[i]);
    }
    long long ans = sum;
    for (i = 0; i < n; i++) {
        sum -= abs(nums1[i] - nums2[i]);
        auto it1 = less.lower_bound(nums2[i]);
        if (it1 != less.end()) {
            ans = min(ans, sum + abs(*it1 - nums2[i]));
        }
        auto it2 = greater.lower_bound(nums2[i]);
        if (it2 != greater.end()) {
            ans = min(ans, sum + abs(*it2 - nums2[i]));
        }
        // 还原
        sum += abs(nums1[i] - nums2[i]);
    }
    return ans % 1000000007;
}


// LC2002
int maxProduct(string s)
{
    // 一共有 2^n - 1 种情况
    int i, j, k;
    int tmp, idx;
    int n = s.size();
    vector<int> status(n);
    vector<int> idxes;
    vector<pair<string, vector<int>>> palindromes;
    string t;
    for (k = 1; k < pow(2, n); k++) {
        tmp = k;
        idx = 0;
        while (tmp) {
            status[idx] = tmp % 2;
            tmp /= 2;
            idx++;
        }
        t.clear();
        idxes.clear();
        for (i = 0; i < n; i++) {
            if (status[i]) {
                t += s[i];
                idxes.emplace_back(i);
            }
        }
        if (IsPalindrome(t)) {
            palindromes.push_back({t, idxes});
        }
    }

    int p, q;
    int ans;
    bool findSameIndex = false;

    ans = 0;
    n = palindromes.size();
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            p = 0;
            q = 0;
            findSameIndex = false;
            while (p < palindromes[i].second.size() && q < palindromes[j].second.size()) {
                if (palindromes[i].second[p] < palindromes[j].second[q]) {
                    p++;
                } else if (palindromes[i].second[p] > palindromes[j].second[q]) {
                    q++;
                } else {
                    findSameIndex = true;
                    break;
                }
            }
            if (!findSameIndex) {
                tmp = palindromes[i].first.size() * palindromes[j].first.size();
                ans = max(ans, tmp);
            }
        }
    }
    return ans;
}


// LC1898
int maximumRemovals(string s, string p, vector<int>& removable)
{
    int i, j, k;
    int left, right, mid;
    string t, tt;
    bool f = false;

    left = 0;
    right = removable.size();

    while (left <= right) {
        mid = (right - left) / 2 + left;
        t = s;
        for (k = 0; k < mid; k++) {
            t[removable[k]] = '0';
        }
        tt.clear();
        for (auto ch : t) {
            if (ch != '0') {
                tt += ch;
            }
        }
        f = false;
        i = j = 0;
        while (i < tt.size() && j < p.size()) {
            if (tt[i] == p[j]) {
                i++;
                j++;
                if (j == p.size()) {
                    f = true;
                    break;
                }
            } else {
                i++;
            }
        }
        if (f) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC2454
vector<int> secondGreaterElement(vector<int>& nums)
{
    // 比一个数大的下一个的下一个, 1 -> 3 -> 2, 则是2
    // 单调递减栈
    int i, idx;
    int n = nums.size();
    unordered_map<int, int> nextGreaterIdx; // 下标 - 下一个更大下标
    set<int> s;
    stack<int> stRight;

    for (i = 0; i < n; i++) {
        if (stRight.empty()) {
            stRight.push(i);
            continue;
        }
        idx = stRight.top();
        if (nums[idx] >= nums[i]) {
            stRight.push(i);
        } else {
            while (nums[idx] < nums[i]) {
                if (nextGreaterIdx.count(idx) == 0) {
                    s.emplace(idx);
                    nextGreaterIdx[idx] = -1;
                } else {
                    nextGreaterIdx[idx] = i;
                }
                stRight.pop();
                if (stRight.empty()) {
                    break;
                }
                idx = stRight.top();
            }
            stRight.push(i);
            for (auto it : s) {
                stRight.push(it);
            }
            s.clear();
        }
    }

    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        if (nextGreaterIdx.count(i) && nextGreaterIdx[i] != -1) {
            ans[i] = nums[nextGreaterIdx[i]];
        } else {
            ans[i] = -1;
        }
    }
    return ans;
}


// LC1092
string shortestCommonSupersequence(string str1, string str2)
{
    int i, j;
    int m = str1.size();
    int n = str2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (str1[i] == str2[j]) {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            } else {
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }
    // 求最长子序列
    string lcs;
    function<void (int, int)> print_lcs = [&dp, &str1, &str2, &lcs, &print_lcs](int i, int j) {
        if(i == 0 || j == 0) {
            return;
        }
        //相等则添加
        if (str1[i - 1] == str2[j - 1]) {
            print_lcs(i - 1, j - 1);
            lcs += str1[i - 1];
        } else if(dp[i - 1][j] >= dp[i][j - 1]) {
            print_lcs(i - 1, j);
        } else {
            print_lcs(i, j - 1);
        }
    };
    print_lcs(m, n);

    // cout << "lcs = " << lcs << endl;

    int cur;
    int size = lcs.size();
    string ans;
    vector<string> part1, part2;

    j = 0;
    cur = 0;
    for (i = 0; i < m; i++) {
        if (str1[i] == lcs[j]) {
            part1.emplace_back(str1.substr(cur, i - cur));
            cur = i + 1;
            j++;
            if (j == size) {
                break;
            }
        }
    }
    if (cur != m) {
        part1.emplace_back(str1.substr(cur));
    }
    j = 0;
    cur = 0;
    for (i = 0; i < n; i++) {
        if (str2[i] == lcs[j]) {
            part2.emplace_back(str2.substr(cur, i - cur));
            cur = i + 1;
            j++;
            if (j == size) {
                break;
            }
        }
    }
    if (cur != n) {
        part2.emplace_back(str2.substr(cur));
    }
    for (i = 0; i < size; i++) {
        ans += part1[i] + part2[i] + lcs[i];
    }
    if (i < part1.size()) {
        ans += part1[part1.size() - 1];
    }
    if (i < part2.size()) {
        ans += part2[part2.size() - 1];
    }
    return ans;
}


// LC1722
int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps)
{
    int i, k;
    int n = source.size();
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, int> sourceNum;
    unordered_map<int, int> targetNum;
    vector<bool> visited(n, false);
    vector<int> idx;

    for (auto as : allowedSwaps) {
        edges[as[0]].emplace(as[1]);
        edges[as[1]].emplace(as[0]);
    }

    function<void (int)> DFS = [&DFS, &visited, &edges, &idx](int cur) {
        visited[cur] = true;
        idx.emplace_back(cur);
        for (auto it : edges[cur]) {
            if (visited[it] == false) {
                DFS(it);
            }
        }
    };
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (visited[i] == false) {
            idx.clear();
            DFS(i);
            sourceNum.clear();
            targetNum.clear();
            for (auto id : idx) {
                sourceNum[source[id]]++;
                targetNum[target[id]]++;
            }
            for (auto it : sourceNum) {
                for (k = 0; k < it.second; k++) {
                    if (targetNum.count(it.first)) {
                        if (targetNum[it.first] == 1) {
                            targetNum.erase(it.first);
                        } else {
                            targetNum[it.first]--;
                        }
                    } else {
                        ans++;
                    }
                }
            }
        }
    }
    return ans;
}


// LC1937
long long maxPoints(vector<vector<int>>& points)
{
    int i, j;
    int m = points.size();
    int n = points[0].size();
    vector<vector<long long>> dp(m, vector<long long>(n, 0)); // dp[i][j] - 在points[i][j]能得到的最大分数

    // dp[i][j] = max{f[i − 1][j′] − abs(j − j′)} + points[i][j] ->
    // max{f[i − 1][j′] − j′} + points[i][j] + j (j < j') or  右边
    // max{f[i − 1][j′] + j′} + points[i][j] - j (j >= j')  左边
    // 用两个数组分别记录 f[i − 1][j′] - j′ 和 f[i − 1][j′] + j′的最大值
    vector<vector<long long>> left(m, vector<long long>(n, 0));
    vector<vector<long long>> right(m, vector<long long>(n, 0));
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0) {
                dp[i][j] = points[i][j];
                continue;
            }
            dp[i][j] = max(left[i - 1][j] + points[i][j] - j, right[i - 1][j] + points[i][j] + j);
        }
        left[i][0] = dp[i][0] + 0; 
        for (j = 1; j < n; j++) {
            left[i][j] = dp[i][j] + j > left[i][j - 1] ? dp[i][j] + j : left[i][j - 1];
        }
        right[i][n - 1] = dp[i][n - 1] - (n - 1);
        for (j = n - 2; j >= 0; j--) {
            right[i][j] = dp[i][j] - j > right[i][j + 1] ? dp[i][j] - j : right[i][j + 1];
        }
    }
    return *max_element(dp[m - 1].begin(), dp[m - 1].end());
}


// LC871
int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations)
{
    int i;
    int n;
    int location, leftFuel;
    int ans;
    priority_queue<int, vector<int>> pq;

    leftFuel = startFuel;
    n = stations.size();
    ans = 0;
    location = 0;
    for (i = 0; i < n; i++) {
        if (stations[i][0] - location > leftFuel) {
            while (stations[i][0] - location > leftFuel) {
                if (pq.empty()) {
                    return -1;
                }
                auto p = pq.top();
                pq.pop();
                ans++;
                leftFuel += p;
            }
            
        }
        leftFuel -= stations[i][0] - location;
        pq.push(stations[i][1]);
        location = stations[i][0];
    }

    while (target - location > leftFuel) {
        if (pq.empty()) {
            return -1;
        }
        auto p = pq.top();
        pq.pop();
        leftFuel += p;
        ans++;
    }
    return ans;
}


// LC294
unordered_map<string, bool> canWinStatus; // 字符串先手是否必胜 例如staus["++"] = true
bool canWin(string s)
{
    int i;
    int n = s.size();
    if (canWinStatus.count(s)) {
        return canWinStatus[s];
    }

    for (i = 0; i < n - 1; i++) {
        if (s[i] == s[i + 1] && s[i] == '+') {
            s[i] = '-';
            s[i + 1] = '-';
            if (canWin(s) == false) { // 翻转后的s仍可以获胜,说明后手会赢,反之先手赢
                return true;
            }
            s[i] = '+';
            s[i + 1] = '+';
        }
    }
    canWinStatus[s] = false; // 循环完到此处说明s不能先手赢,否则在循环中就直接return了
    return canWinStatus[s];
}


// LC464
// 数字可以重复取的情况
bool canIWin1(int maxChoosableInteger, int desiredTotal)
{
    if (maxChoosableInteger >= desiredTotal) {
        return true;
    }

    int i, j;
    vector<int> dp(desiredTotal + 1, false); // dp[i] - 剩余i先手能否赢

    for (i = 1; i <= maxChoosableInteger; i++) {
        dp[i] = true;
    }
    for (i = maxChoosableInteger + 1; i <= desiredTotal; i++) {
        for (j = 1; j <= maxChoosableInteger; j++) {
            if (dp[i - j] == false) {
                dp[i] = true;
                break;
            }
        }
        if (j == maxChoosableInteger + 1) {
            dp[i] = false;
        }
    }
    return dp[desiredTotal];
}
// 不可以重复取
bool canIWin(int maxChoosableInteger, int desiredTotal)
{
    if (maxChoosableInteger >= desiredTotal) {
        return true;
    }

    int i;
    int sum = 0;
    for (i = 1; i <= maxChoosableInteger; i++) {
        sum += i;
    }
    if (sum < desiredTotal) {
        return false;
    }

    unordered_map<int, bool> status; // 表示当前取数的状态
    function<bool (int, int, int, int)> DFS = [&status, &DFS](int usedNum, int cur, int maxChoosableInteger, int desiredTotal) {
        if (status.count(usedNum)) {
            return status[usedNum];
        }
        int i;
        for (i = maxChoosableInteger; i >= 1; i--) {
            if ((usedNum & (1 << i)) == 0) {
                if (cur + i >= desiredTotal) {
                    status[usedNum] = true;
                    return true;
                }
                if (DFS(usedNum + (1 << i), cur + i, maxChoosableInteger, desiredTotal) == false) {
                    status[usedNum] = true;
                    return true;
                }
            }
        }
        status[usedNum] = false;
        return false;
    };
    int usedNum = 0;
    return DFS(usedNum, 0, maxChoosableInteger, desiredTotal);
}


// LC375
int getMoneyAmount(int n)
{
    // dp[i][j] - 区间[i, j]猜到数字的最小花费
    int i, j, k;
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0x3f3f3f3f));

    for (j = 1; j <= n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 0;
            } else if (i + 1 == j) {
                dp[i][j] = min(i, j);
            } else {
                // 假设所猜数字是k
                for (k = i + 1; k <= j - 1; k++) {
                    dp[i][j] = min(dp[i][j], max(dp[k + 1][j], dp[i][k - 1]) + k);
                }
            }
        }
    }
    return dp[1][n];
}


// LC1871
// BFS超时
bool canReach1(string s, int minJump, int maxJump)
{
    int i;
    int n = s.size();
    vector<bool> visited(n, false);

    if (s[n - 1] == '1') {
        return false;
    }
    vector<int> idx;
    for (i = 1; i < n; i++) {
        if (s[i] == '0') {
            idx.emplace_back(i);
        }
    }
    vector<int> idx1 = idx;
    reverse(idx1.begin(), idx1.end());
    queue<int> q;
    q.push(0);
    while (!q.empty()) {
        auto t = q.front();
        q.pop();

        visited[t] = true;
        if (t == n - 1) {
            return true;
        }
        auto left = lower_bound(idx.begin(), idx.end(), t + minJump);
        auto right = lower_bound(idx1.begin(), idx1.end(), t + maxJump, greater<int>());
        if (left != idx.end() && right != idx1.end()) {
            for (i = left - idx.begin(); i <= right - idx1.begin(); i++) {
                if (!visited[idx[i]]) {
                    q.push(idx[i]);
                }
            }
        }
    }
    return false;
}
bool canReach(string s, int minJump, int maxJump)
{
    int i;
    int n = s.size();
    int left, right;
    vector<int> prefixSum(n);
    vector<bool> dp(n, false); // dp[i] - 第i位能否到达

    if (s[n - 1] == '1') {
        return false;
    }

    for (i = 0; i < minJump; i++) {
        prefixSum[i] = 1;
    }
    // 同步更新前缀和
    for (i = minJump; i < n; i++) {
        if (s[i] == '0') {
            left = i - maxJump;
            right = i - minJump;
            if (right == 0) {
                dp[i] = true;
                prefixSum[i] = prefixSum[i - 1] + 1;
                continue;
            }
            if (left < 0) {
                left = 0;
            }
            if (left == 0) {
                if (prefixSum[right] > 0) {
                    prefixSum[i] = prefixSum[i - 1] + 1;
                    dp[i] = true;
                }
            } else {
                if (prefixSum[right] - prefixSum[left - 1] > 0) {
                    prefixSum[i] = prefixSum[i - 1] + 1;
                    dp[i] = true;
                }
            }
            if (dp[i] == false) {
                prefixSum[i] = prefixSum[i - 1];
            }
        } else {
            prefixSum[i] = prefixSum[i - 1];
        }
    }
    return dp[n - 1];
}


// LC2964
int divisibleTripletCount(vector<int>& nums, int d)
{
    int i, j;
    int a, b;
    int ans;
    int left, right, mid;
    int n = nums.size();
    unordered_map<int, vector<int>> um;

    for (i = 0; i < n; i++) {
        um[nums[i] % d].emplace_back(i);
    }
    ans = 0;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            a = (nums[i] + nums[j]) % d;
            b = (d - a) % d;
            vector<int> v = um[b];
            left = 0;
            right = v.size() - 1;
            // 所求为left;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (v[mid] <= j) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            ans += v.size() - left;
        }
    }
    return ans;
}


// LC2967
void createPalindrome(string& num, int index, set<int>& resultLess, 
    set<int, greater<>>& resultGreater)
{
    int i;
    int n = num.size();

    if ((n % 2 == 0 && index >= n / 2) || (n % 2 == 1 && index > n / 2)) {
        resultLess.emplace(atoi(num.c_str()));
        resultGreater.emplace(atoi(num.c_str()));
        return;
    }
    for (i = 0; i <= 9; i++) {
        if (index == 0 && i == 0 ) {
            continue;
        }

        num[index] = i + '0';
        num[n - 1 - index] = num[index];
        createPalindrome(num, index + 1, resultLess, resultGreater);
    }
}
long long minimumCost(vector<int>& nums)
{
    sort(nums.begin(), nums.end());

    int i;
    int k;
    int n = nums.size();
    int target;
    long long ans = 0;

    k = n / 2;
    string t = to_string(nums[k]);
    if (IsPalindrome(t)) {
        for (i = 0; i < n; i++) {
            ans += abs(nums[i] - nums[k]);
        }
        return ans;
    }

    set<int> resultLess;
    set<int, greater<>> resultGreater;
    string num = t;
    createPalindrome(num, 0, resultLess, resultGreater);

    // 第一个大于等于nums[k]的回文数
    target = *resultLess.lower_bound(nums[k]);
    for (i = 0; i < n; i++) {
        ans += abs(nums[i] - target);
    }

    // 第一个小于等于nums[k]的回文数
    long long tmp = 0;
    auto it = resultGreater.lower_bound(nums[k]);
    if (it == resultGreater.end()) {
        target = nums[k] - 1;
    } else {
        target = *it;
    }
    for (i = 0; i < n; i++) {
        tmp += abs(nums[i] - target);
    }
    return min(ans, tmp);
}


// LC2968
int maxFrequencyScore(vector<int>& nums, long long k)
{
    sort(nums.begin(), nums.end());

    int i;
    int n = nums.size();
    int left, right, mid;
    long long median; // 中位数
    long long need;
    vector<long long> prefixSum(n);

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
    }

    left = 1;
    right = nums.size();
    while (left <= right) {
        mid = (right - left) / 2 + left;
        need = 0;
        // 滑动窗口
        for (i = 0; i <= n - mid; i++) {
            if (mid % 2 == 1) {
                median = nums[(i + i + mid - 1) / 2];
            } else {
                median = (nums[(i + i + mid - 1) / 2] + nums[(i + i + mid - 1) / 2 + 1]) / 2;
            }
            if (i == 0) {
                need = (mid  + 1) / 2 * median - prefixSum[(i + i + mid - 1) / 2] + 
                    prefixSum[i + mid - 1] - prefixSum[(i + i + mid - 1) / 2] - mid / 2 * median;
            } else {
                need = (mid  + 1) / 2 * median - (prefixSum[(i + i + mid - 1) / 2]  - prefixSum[i - 1]) + 
                    prefixSum[i + mid - 1] - prefixSum[(i + i + mid - 1) / 2] - mid / 2 * median;
            }
            // printf ("i = %d, median = %d, need = %d, mid = %d\n", i, median, need, mid);
            if (need <= k) {
                left = mid + 1;
                break;
            }
        }
        if (i == n - mid + 1) {
            right = mid - 1;
        }
    }
    return right;
}


// LC2276
// to do
class CountIntervals {
public:
    set<vector<int>> s;
    int cnt;
    CountIntervals()
    {
        cnt = 0;
    }
    
    void add(int left, int right)
    {
        if (s.empty()) {
            s.insert({left, right});
            cnt += right - left + 1;
            return;
        }
        auto it = s.lower_bound({left, right});
    }
    
    int count()
    {
        return cnt;
    }
};


// LC458
// N进制编码问题
int poorPigs(int buckets, int minutesToDie, int minutesToTest)
{
    // 有n次实验机会, 就转换成n + 1进制
    int N = minutesToTest / minutesToDie + 1;
    int t;
    int cnt = 0;

    while (t) {
        cnt++;
        t /= N;
    }
    if (buckets == pow(2, cnt - 1)) {
        cnt--;
    }
    return cnt;
}


// LC1901
vector<int> findPeakGrid(vector<vector<int>>& mat)
{
    int m = mat.size();
    int n = mat[0].size();
    int left, right, mid;
    int j;

    left = 0;
    right = m - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        j = max_element(mat[mid].begin(), mat[mid].end()) - mat[mid].begin();
        if (mid > 0 && mat[mid -1][j] > mat[mid][j]) {
            right = mid - 1;
        } else if (mid < m - 1 && mat[mid + 1][j] > mat[mid][j]) {
            left = mid + 1;
        } else {
            break;
        }
    }
    return {mid, j};
}


// LC1359
int countOrders(int n)
{
    // 一对一对pd的增加
    int mod = 1000000007;
    int i;
    vector<long long> dp(n + 1);

    dp[1] = 1;
    for (i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] * ((2 * (i - 1) + 1) + 1) * (2 * i - 1) / 2 % mod;
    }
    return dp[n];
}


// LC1220
int countVowelPermutation(int n)
{
    int mod = 1000000007;
    int i;

    vector<vector<long long>> dp(n, vector<long long>(5));
    // dp[i][0] - 第i位以'a'结尾的序列数目

    for (i = 0; i < 5; i++) {
        dp[0][i] = 1;
    }
    for (i = 1; i < n; i++) {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        dp[i][0] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][4]) % mod; // [e, i, u] a
        dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % mod; // [a, i] e
        dp[i][2] = (dp[i - 1][1] + dp[i - 1][3]) % mod; // [e, o] i
        dp[i][3] = dp[i - 1][2]; // [i] o
        dp[i][4] = (dp[i - 1][2] + dp[i - 1][3]) % mod; // [i, o] u
    }
    long long ans = 0;
    for (auto it : dp[n - 1]) {
        ans = (ans + it) % mod;
    }
    return ans;
}


// LC1111
vector<int> maxDepthAfterSplit(string seq)
{
    stack<pair<int, int>> st;
    int i;
    int n = seq.size();
    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push({i, 0});
            continue;
        }
        if (seq[i] == ')') {
            auto p = st.top();
            st.pop();
            ans[p.first] = p.second;
            ans[i] = p.second;
        } else {
            auto p = st.top();
            st.push({i, 1 - p.second});
        }
    }
    return ans;
}


// LC2311
int longestSubsequence(string s, int k)
{
    int i, j;
    int n = s.size();
    long long ans;
    long long t;
    bool flag = false;
    // dp[i] - 以s[i]结尾且表示不超过k的最长子序列
    vector<long long> dp(n, 0);

    dp[0] = 1;
    ans = 1;
    for (i = 1; i < n; i++) {
        dp[i] = 1;
        t = s[i] - '0';
        flag = false;
        for (j = i - 1; j >= 0; j--) {
            if (s[j] == '0') {
                dp[i]++;
            } else {
                if (flag) {
                    continue;
                }
                if (i - j > 30) {
                    flag = true;
                    continue;
                }
                t += 1ll << (i - j);
                if (t > k) {
                    flag = true;
                } else {
                    dp[i]++;
                }
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC852
int peakIndexInMountainArray(vector<int>& arr)
{
    // 山脉数组
    int left, right, mid;
    int n = arr.size();

    left = 0;
    right = n - 1;

    while (left < right) {
        mid = (right - left) / 2 + left;
        if (mid < n - 1 && arr[mid] > arr[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2832
vector<int> maximumLengthOfRanges(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int top, topIdx;
    vector<int> ans(n, 1);
    stack<int> st;

    // 右边界 - 单调递减栈
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        topIdx = st.top();
        top = nums[topIdx];
        if (top >= nums[i]) {
            st.push(i);
            continue;
        }
        while (top < nums[i]) {
            ans[topIdx] += i - topIdx - 1;
            st.pop();
            if (st.empty()) {
                break;
            }
            topIdx = st.top();
            top = nums[topIdx];
        }
        st.push(i);
    }
    while (!st.empty()) {
        topIdx = st.top();
        st.pop();
        ans[topIdx] += n - 1 - topIdx;
    }

    // 左边界 - 单调递减栈
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        topIdx = st.top();
        top = nums[topIdx];
        if (top >= nums[i]) {
            st.push(i);
            continue;
        }
        while (top < nums[i]) {
            ans[topIdx] += topIdx - i - 1;
            st.pop();
            if (st.empty()) {
                break;
            }
            topIdx = st.top();
            top = nums[topIdx];
        }
        st.push(i);
    }
    while (!st.empty()) {
        topIdx = st.top();
        st.pop();
        ans[topIdx] += topIdx;
    }
    return ans;
}


// LC1478
int minDistance(vector<int>& houses, int k)
{
    int i, j, p;
    int mid;
    int n = houses.size();
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, INT_MAX));

    // 预处理 - 任意两个下标点有一个邮箱的距离和
    vector<vector<long long>> dist(n, vector<long long>(n, 0));
    sort(houses.begin(), houses.end());
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if ((j - i) % 2 == 0) {
                mid = houses[(i + j) / 2];
            } else {
                mid = (houses[(i + j) / 2] + houses[(i + j + 1) / 2]) / 2;
            }
            for (p = i; p <= j; p++) {
                dist[i][j] += abs(houses[p] - mid);
            }
        }
    }
    for (j = 1; j <= k; j++) {
        for (i = 1; i <= n; i++) {
            if (j >= i) {
                dp[i][j] = 0;
                continue;
            }
            if (j == 1) {
                dp[i][j] = dist[0][i - 1];
            } else {
                for (p = 1; p < i; p++) {
                    dp[i][j] = min(dp[i][j], dp[p][j - 1] + dist[p][i - 1]);
                }
            }
        }
    }
    return dp[n][k];
}


// LC1477
int minSumOfLengths(vector<int>& arr, int target)
{
    int i;
    int n = arr.size();
    unordered_map<int, int> sum;
    vector<int> prefixSum(n, 0);
    vector<vector<int>> range;
    vector<int> minRangeDiff;

    prefixSum[0] = arr[0];
    sum[arr[0]] = 0;
    if (arr[0] == target) {
        range.push_back({0, 0});
    }
    for (i = 1; i < n; i++) {
        prefixSum[i] = arr[i] + prefixSum[i - 1];
        sum[prefixSum[i]] = i;
        if (prefixSum[i] == target) {
            range.push_back({0, i});
        }
        if (sum.count(prefixSum[i] - target)) {
            range.push_back({sum[prefixSum[i] - target] + 1, i});
        }
    }
    if (range.size() < 2) {
        return -1;
    }

    n = range.size();
    minRangeDiff.resize(n);
    minRangeDiff[0] = range[0][1] - range[0][0] + 1;
    for (i = 1; i < n; i++) {
        auto t = range[i][1] - range[i][0] + 1;
        if (t < minRangeDiff[i - 1]) {
            minRangeDiff[i] = t;
        } else {
            minRangeDiff[i] = minRangeDiff[i - 1];
        }
    }

    int left, right, mid;
    int ans = 0x3f3f3f3f;
    for (i = 1; i < n; i++) {
        left = 0;
        right = i - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) /2 + left;
            if (range[mid][1] < range[i][0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            continue;
        }
        ans = min(ans, range[i][1] - range[i][0] + 1 + minRangeDiff[right]);
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC2971
long long largestPerimeter(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left, right, mid;

    sort(nums.begin(), nums.end());

    long long sum = 0;
    long long ans = -1;
    for (i = 0; i < n - 1; i++) {
        sum += nums[i];
        if (i < 1) {
            continue;
        }
        left = i + 1;
        right = n - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) /2 + left;
            if (sum > nums[mid]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right <= i) {
            continue;
        }
        ans = max(ans, sum + nums[right]);
    }
    return ans;
}


// LC2972
long long incremovableSubarrayCount(vector<int>& nums)
{
    int i;
    int n = nums.size();
    long long ans;
    int len = 1;
    
    vector<int> front, rear;
    front.emplace_back(nums[0]);
    for (i = 1; i < n; i++) {
        if (nums[i] > nums[i - 1]) {
            front.emplace_back(nums[i]);
            len++;
        } else {
            break;
        }
    }
    if (len == n) {
        return (long long)n * (n + 1) / 2;
    }
    ans = 1;
    ans += len;

    len = 1;
    rear.emplace_back(nums[n - 1]);
    for (i = n - 2; i >= 0; i--) {
        if (nums[i + 1] > nums[i]) {
            rear.emplace_back(nums[i]);
            len++;
        } else {
            break;
        }
    }
    ans += len;

    n = front.size();
    int left, right, mid;
    // cout << ans << endl;
    reverse(rear.begin(), rear.end());
    for (i = 0; i < n; i++) {
        left = 0;
        right = rear.size() - 1;
        // 所求为left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (rear[mid] < front[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left >= rear.size()) {
            continue;
        }
        // cout << left << endl;
        if (rear[left] == front[i]) {
            ans += rear.size() - left - 1;
        } else {
            ans += rear.size() - left;
        }
    }

    return ans;
}


// LC2975
int maximizeSquareArea(int m, int n, vector<int>& hFences, vector<int>& vFences)
{
    sort(hFences.begin(), hFences.end());
    sort(vFences.begin(), vFences.end());

    int i, j;
    long long ans = -1;
    vector<long long> hvector;
    vector<long long> vvector;
    unordered_set<long long> diffHSet;

    hvector.emplace_back(1);
    for (auto h : hFences) {
        hvector.emplace_back(h);
    }
    hvector.emplace_back(m);

    vvector.emplace_back(1);
    for (auto v : vFences) {
        vvector.emplace_back(v);
    }
    vvector.emplace_back(n);

    for (i = 0; i < hvector.size() - 1; i++) {
        for (j = i + 1; j < hvector.size(); j++) {
            diffHSet.emplace(hvector[j] - hvector[i]);
        }
    }
    // cout << diffHSet.size() << endl;
    for (i = 0; i < vvector.size() - 1; i++) {
        for (j = i + 1; j < vvector.size(); j++) {
            if (diffHSet.count(vvector[j] - vvector[i])) {
                ans = max(ans, (vvector[j] - vvector[i]) * (vvector[j] - vvector[i]));
            }
        }
    }
    return ans > 0 ? ans % 1000000007 : ans;
}


// LC924 - 删除某点不删除其网络
// LC928 - 删除某点同时也删除其网络结构
int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial)
{
    int i, k;
    int n = graph.size();
    int kind = initial.size();
    int minNode = INT_MAX;
    int cnt, ans;
    unordered_set<int> visited;
    queue<int> q;

    sort(initial.begin(), initial.end());
    for (k = 0; k < kind; k++) {
        for (i = 0; i < kind; i++) {
            if (i != k) {
                q.push(initial[i]);
            }
        }
        cnt = 0;
        visited.clear();
        while (!q.empty()) {
            auto t = q.front();
            q.pop();

            if (visited.count(t)) {
                continue;
            }
            cnt++;
            visited.emplace(t);
            for (i = 0; i < n; i++) {
                // LC924
                // if (graph[t][i] == 1 && i != t && visited.count(i) == 0) {
                // LC928
                if (graph[t][i] == 1 && i != t && i != initial[k] && visited.count(i) == 0) {
                    q.push(i);
                }
            }
        }
        if (cnt < minNode) {
            minNode = cnt;
            ans = initial[k];
        }
    }
    return ans;
}


// LC2296
// 暴力,超时
class TextEditor1 {
public:
    TextEditor1()
    {
        cursorPos = 0;
    }

    void addText(string text)
    {
        if (this->text.empty()) {
            this->text = text;
            cursorPos = text.size();
            return;
        }
        if (cursorPos == this->text.size()) {
            this->text += text;
            cursorPos += text.size();
            return;
        } else if (cursorPos == 0) {
            this->text = text + this->text;
            cursorPos += text.size();
            return;
        }
        int len = this->text.size();
        string front = this->text.substr(0, cursorPos);
        string rear = this->text.substr(cursorPos);
        this->text = front + text + rear;
        cursorPos += text.size();
    }

    int deleteText(int k)
    {
        int ans;
        if (k >= cursorPos) {
            ans = cursorPos;
            this->text = this->text.substr(cursorPos);
            cursorPos = 0;
            return ans;
        }
        string front = this->text.substr(0, cursorPos - k);
        string rear = this->text.substr(cursorPos);
        this->text = front + rear;
        cursorPos -= k;
        return k;
    }

    string cursorLeft(int k)
    {
        if (k >= cursorPos) {
            cursorPos = 0;
            return "";
        }
        cursorPos -= k;
        if (cursorPos <= 10) {
            return this->text.substr(0, cursorPos);
        }
        return this->text.substr(cursorPos - 10, 10);
    }

    string cursorRight(int k)
    {
        if (k + cursorPos >= this->text.size()) {
            cursorPos = this->text.size();
            if (cursorPos <= 10) {
                return this->text.substr(0, cursorPos);
            }
            return this->text.substr(cursorPos - 10, 10);
        }
        cursorPos += k;
        if (cursorPos <= 10) {
            return this->text.substr(0, cursorPos);
        }
        return this->text.substr(cursorPos - 10, 10);
    }
private:
    int cursorPos;
    string text;
};
// 双向链表
class TextEditor {
public:
    struct ListNode {
        char val;
        ListNode *prev;
        ListNode *next;
        ListNode(char v) : val(v), prev(nullptr), next(nullptr) {};
    };
    TextEditor()
    {
        head = new ListNode('@');
        tail = new ListNode('#');
        head->next = tail;
        tail->prev = head;
        cursor = tail;
    }
    ~TextEditor()
    {
        ListNode *t = head;
        ListNode *cur = t;
        while (t != tail) {
            t = t->next;
            delete cur;
            cur = t;
        }
        delete cur;
    }

    void addText(string text)
    {
        for (auto ch : text) {
            ListNode *node = new ListNode(ch);
            auto prevNode = cursor->prev;
            prevNode->next = node;
            node->prev = prevNode;
            node->next = cursor;
            cursor->prev = node;
        }
    }

    int deleteText(int k)
    {
        int i;
        int ans = 0;

        for (i = 0; i < k; i++) {
            auto prevNode = cursor->prev; // 要删的点
            if (prevNode == head) {
                break;
            }
            auto prevprevNode = prevNode->prev;
            delete prevNode;
            prevprevNode->next = cursor;
            cursor->prev = prevprevNode;
            ans++;
        }
        return ans;
    }

    string cursorLeft(int k)
    {
        int i;
        while (k) {
            cursor = cursor->prev;
            if (cursor == head) {
                cursor = head->next;
                return "";
            }
            k--;
        }
        ListNode *point = cursor->prev;
        string ans;
        for (i = 0; i < 10; i++) {
            if (point == head) {
                break;
            }
            ans += point->val;
            point = point->prev;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }

    string cursorRight(int k)
    {
        int i;
        while (k) {
            if (cursor == tail) {
                break;
            }
            cursor = cursor->next;
            k--;
        }
        ListNode *point = cursor->prev;
        string ans;
        for (i = 0; i < 10; i++) {
            if (point == head) {
                break;
            }
            ans += point->val;
            point = point->prev;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
private:
    ListNode *head;
    ListNode *tail;
    ListNode *cursor;
};


// LC87
bool isScramble(string s1, string s2)
{
    int i, j, k, p;
    int len = s1.size();
    bool f1, f2;

    // dp[i][j][k] - s1以i下标开始, s2以j下标开始长度为k的子字符串是否可以相互转化
    // 所求dp[0][0][len];
    vector<vector<vector<bool>>> dp(len, vector<vector<bool>>(len, vector<bool>(len + 1, false)));

    for (k = 1; k <= len; k++) {
        for (i = 0; i < len; i++) {
            for (j = 0; j < len; j++) {
                if (i + k > len || j + k > len) {
                    break;
                }
                if (k == 1) {
                    dp[i][j][k] = s1[i] == s2[j];
                    continue;
                }
                for (p = 1; p < k; p++) {
                    f1 = dp[i][j][p] && dp[i + p][j + p][k - p]; // 不交换
                    f2 = dp[i][k + j - p][p] && dp[i + p][j][k - p]; // 交换
                    dp[i][j][k] = dp[i][j][k] || f1 || f2;
                }
            }
        }
    }
    return dp[0][0][len];
}


// LC305
vector<int> numIslands2(int m, int n, vector<vector<int>>& positions)
{
    int k;
    int no, land, oriLand;
    int nrow, ncol, p, np;
    unordered_map<int, vector<int>> lands;
    unordered_map<int, int> posNo;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<int> ans;

    no = 1;
    for (auto pos : positions) {
        p = pos[0] * n + pos[1];
        for (k = 0; k < 4; k++) {
            nrow = pos[0] + directions[k][0];
            ncol = pos[1] + directions[k][1];
            if (nrow < 0 || nrow >= m || ncol < 0 || ncol >= n) {
                continue;
            }
            np = nrow * n + ncol;
            if (posNo.count(np)) {
                if (posNo.count(p) == 0) {
                    land = posNo[np];
                    lands[land].emplace_back(p);
                    posNo[p] = land;
                } else {
                    if (posNo[p] == posNo[np]) {
                        continue;
                    }
                    land = posNo[np];
                    auto v = lands[land];
                    oriLand = posNo[p];
                    for (auto vv : v) {
                        lands[oriLand].emplace_back(vv);
                        posNo[vv] = oriLand;
                    }
                    lands.erase(land);
                }
            }
        }
        if (posNo.count(p) == 0) {
            posNo[p] = no;
            lands[no].emplace_back(p);
            no++;
        }
        ans.emplace_back(lands.size());
    }
    return ans;
}


// LC403
bool canCross(vector<int>& stones)
{
    if (stones[1] - stones[0] != 1) {
        return false;
    }

    int i, j;
    int n = stones.size();
    int dist;
    // dp[i][j] - 青蛙是否能从j跳到i
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    unordered_map<int, unordered_set<int>> steps; // 跳到i的所有步距
    dp[1][0] = true;
    steps[1].emplace(1);
    for (i = 2; i < n; i++) {
        for (j = i - 1; j >= 1; j--) {
            dist = stones[i] - stones[j];
            if (steps[j].count(dist) || steps[j].count(dist - 1) || steps[j].count(dist + 1)) {
                dp[i][j] = true;
                steps[i].emplace(dist);
            }
        }
    }
    for (auto f : dp[n - 1]) {
        if (f) {
            return true;
        }
    }
    return false;
}


// LC1235
int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit)
{
    int i;
    int n = startTime.size();
    int ans;
    int left, right, mid;
    vector<vector<int>> range;
    vector<int> prefixMax(n); // 前缀最大值

    for (i = 0; i < n; i++) {
        range.push_back({startTime[i], endTime[i], profit[i]});
    }
    sort(range.begin(), range.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    // dp[i] - 以第i份兼职结束的最大收益
    vector<int> dp(n, 0);

    dp[0] = range[0][2];
    prefixMax[0] = dp[0];
    ans = dp[0];
    for (i = 1; i < n; i++) {
        left = 0;
        right = i - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (range[mid][1] <= range[i][0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            dp[i] = range[i][2];
        } else {
            dp[i] = range[i][2] + prefixMax[right];
        }
        prefixMax[i] = dp[i] > prefixMax[i - 1] ? dp[i] : prefixMax[i - 1];
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC2251
vector<int> fullBloomFlowers(vector<vector<int>>& flowers, vector<int>& people)
{
    int i;
    int n;
    int left, right, mid;
    int a, b;
    vector<int> ans;
    vector<vector<int>> f1, f2;

    f1 = flowers;
    sort(f1.begin(), f1.end());

    f2 = flowers;
    sort(f2.begin(), f2.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    n = people.size();
    for (i = 0; i < n; i++) {
        left = 0;
        right = f1.size() - 1;
        // f1中前区间第一个小于等于people[i]的值, right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (f1[mid][0] > people[i]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        a = right + 1; // 在people[i]时有a朵花开放
        // f2中后区间第一个小于people[i]的值, right
        left = 0;
        right = f2.size() - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (f2[mid][1] >= people[i]) {
                right = mid - 1;
                
            } else {
                left = mid + 1;
            }
        }
        b = right + 1; // 在people[i]时有b朵花凋零
        ans.emplace_back(a - b);
    }
    return ans;
}


// LC1231
int maximizeSweetness(vector<int>& sweetness, int k)
{
    int i;
    int n = sweetness.size();
    int cnt, curSum;
    int left, right, mid;

    left = *min_element(sweetness.begin(), sweetness.end());
    right = accumulate(sweetness.begin(), sweetness.end(), 0);
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curSum = cnt = 0;
        for (i = 0; i < n; i++) {
            curSum += sweetness[i];
            if (curSum >= mid) {
                cnt++;
                curSum = 0;
            }
        }
        if (cnt >= k + 1) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC149
int maxPoints_LC149(vector<vector<int>>& points)
{
    int i, j;
    int n = points.size();
    int a, b;
    unordered_map<int, map<vector<int>, int>> pointK;
    vector<int> k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }
            if (points[i][0] == points[j][0]) {
                pointK[i][{0, 0}]++;
            } else if (points[i][1] == points[j][1]) {
                pointK[i][{0, 1}]++;
            } else {
                a = points[j][1] - points[i][1];
                b = points[j][0] - points[i][0];
                k = {a / gcd(a, b), b / gcd(a, b)};
                pointK[i][k]++;
            }
        }
    }
    int ans = 1;
    for (auto it : pointK) {
        for (auto it1 : it.second) {
            ans = max(ans, it1.second + 1);
        }
    }
    return ans;
}


// LC2271
int maximumWhiteTiles(vector<vector<int>>& tiles, int carpetLen)
{
    int i;
    int n = tiles.size();
    int left, right, mid;
    int ans, t;
    int start;
    vector<int> prefixSum(n, 0);

    sort(tiles.begin(), tiles.end());
    prefixSum[0] = tiles[0][1] - tiles[0][0] + 1;
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + tiles[i][1] - tiles[i][0] + 1;
    }

    ans = 0;
    for (i = 0; i < n; i++) {
        left = 0;
        right = i;
        // 从后往前覆盖, 避免被浪费
        start = tiles[i][1] - carpetLen + 1;
        // 查找第一个比start小的tiles[i][1], 所求为right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (tiles[mid][1] >= start) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (right < 0) {
            t = prefixSum[i];
        } else {
            t = prefixSum[i] - prefixSum[right];
        }
        if (start > tiles[right + 1][0]) {
            t -= start - tiles[right + 1][0];
        }
        ans = max(ans, t);
    }
    return ans;
}


// LC2952
int minimumAddedCoins(vector<int>& coins, int target)
{
    int i;
    int value, idx;
    int n = coins.size();
    int cnt;

    sort(coins.begin(), coins.end());
    value = 1;
    idx = cnt = 0;
    while (value <= target) {
        if (idx == n) {
            value *= 2;
            cnt++;
            continue;
        }
        if (value < coins[idx]) {
            value *= 2;
            cnt++;
        } else {
            value += coins[idx];
            idx++;
        }
    }
    return cnt;
}


// LC2907
int maxProfit_LC2907(vector<int>& prices, vector<int>& profits)
{
    int i, j;
    int n = prices.size();
    int ans;
    // dp[i][k] - 取i商品且一共选k个的最大利润
    vector<vector<int>> dp(n, vector<int>(4, 0));
    for (i = 0; i < n; i++) {
        dp[i][1] = profits[i];
    }
    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (prices[i] > prices[j]) {
                dp[i][2] = max(dp[i][2], profits[i] + dp[j][1]);
            }
        }
    }
    ans = -1;
    for (i = 2; i < n; i++) {
        for (j = i - 1; j >= 1; j--) {
            if (prices[i] > prices[j] && dp[j][2] != 0) {
                dp[i][3] = max(dp[i][3], profits[i] + dp[j][2]);
                ans = max(ans, dp[i][3]);
            }
        }
    }
    return ans;
}


// LC2992
int selfDivisiblePermutationCount(int n)
{
    int ans;
    vector<bool> visited(n + 1, false);
    vector<int> permutation(n + 1);

    function<void (int, int)> CreatePermutation = [&CreatePermutation, &visited, &permutation, &ans](int idx, int n) {

        int i;
        if (idx > n) {
            ans++;
            return;
        }
        for (i = 1; i <= n; i++) {
            if (visited[i] == false && (i % idx == 0 || idx % i == 0)) {
                permutation[idx] = i;
                visited[i] = true;
                CreatePermutation(idx + 1, n);
                visited[i] = false;
            }
        }
    };
    ans = 0;
    CreatePermutation(1, n);
    return ans;
}


// LC639
int numDecodings(string s)
{
    int i;
    int n = s.size();
    int mod = 1000000007;
    vector<long long> dp(n + 1, 0);

    dp[0] = 1;
    if (s[0] == '0') {
        return 0;
    }
    if (s[0] == '*') {
        dp[1] = 9;
    } else {
        dp[1] = 1;
    }
    for (i = 1; i < n; i++) {
        if (s[i - 1] == '*') {
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 15) % mod; // 1 - 9, 11 - 19, 21 - 26
            } else if (s[i] == '0') {
                dp[i + 1] = dp[i - 1] * 2 % mod; // 10, 20
            } else if (s[i] > '6') {
                dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 7 - 9, 17 - 19
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1] * 2) % mod; // 7 - 9, 17 - 19, 27 - 29
            }
        } else if (s[i - 1] == '0' || s[i - 1] > '2') {
            if (s[i] == '*') {
                dp[i + 1] = dp[i] * 9 % mod; // 1 - 9
            } else {
                if (s[i] == '0') {
                    return 0;
                }
                dp[i + 1] = dp[i];
            }
        } else if (s[i - 1] == '2') {
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 6) % mod; // 1 - 9, 21 - 26
            } else {
                if (s[i] > '6') {
                    dp[i + 1] = dp[i]; // 7 - 9
                } else if (s[i] == '0') {
                    dp[i + 1] = dp[i - 1]; // 20
                } else {
                    dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 1 - 6, 21 - 26
                }
            }
        } else { // s[i - 1] == '1'
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 9) % mod; // 1 - 9, 11 - 19
            } else {
                if (s[i] == '0') {
                    dp[i + 1] = dp[i - 1]; // 10
                } else {
                    dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 1 - 9, 11 - 19
                }
            }
        }
    }
    return dp[n];
}


// LC91
int numDecodings_LC91(string s)
{
    int i;
    int n = s.size();
    vector<long long> dp(n + 1, 0);

    dp[0] = 1;
    if (s[0] == '0') {
        return 0;
    }
    dp[1] = 1;
    for (i = 1; i < n; i++) {
        if (s[i - 1] == '0' || s[i - 1] > '2') {
            if (s[i] == '0') {
                    return 0;
            }
            dp[i + 1] = dp[i];
        } else if (s[i - 1] == '2') {
            if (s[i] > '6') {
                dp[i + 1] = dp[i]; // 7 - 9
            } else if (s[i] == '0') {
                dp[i + 1] = dp[i - 1]; // 20
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1]); // 1 - 6, 21 - 26
            }
        } else { // s[i - 1] == '1'
            if (s[i] == '0') {
                dp[i + 1] = dp[i - 1]; // 10
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1]); // 1 - 9, 11 - 19
            }
        }
    }
    return dp[n];
}


// LC1944
vector<int> canSeePersonsCount(vector<int>& heights)
{
    int i;
    int idx;
    int n = heights.size();
    vector<int> ans(n, 0);
    // 单调递减栈
    stack<int> st;

    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        idx = st.top();
        while (heights[i] > heights[idx]) {
            ans[idx]++;
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        if (!st.empty()) {
            ans[idx]++;
        }
        st.push(i);
    }
    return ans;
}


// LC3002
int maximumSetSize(vector<int>& nums1, vector<int>& nums2)
{
    unordered_set<int> s1(nums1.begin(), nums1.end());
    unordered_set<int> s2(nums2.begin(), nums2.end());

    int n1 = s1.size();
    int n2 = s2.size();
    int common = 0; // 相同元素个数
    for (auto it : s1) {
        if (s2.count(it)) {
            common++;
        }
    }
    int unique1 = n1 - common;
    int unique2 = n2 - common;
    int n = nums1.size();
    int m = n / 2;
    int ans = 0;

    if (unique1 >= m) {
        ans += m;
        if (unique2 >= m) {
            ans += m;
        } else {
            ans += min(unique2 + common, m);
        }
    } else {
        if (unique1 + common >= m) {
            ans += m;
            if (unique2 >= m) {
                ans += m;
            } else {
                ans += unique2 + min(common - (m - unique1), m - unique2);
            }
        } else {
            ans += unique1 + common;
            if (unique2 >= m) {
                ans += m;
            } else {
                ans += unique2;
            }
        }
    }

    return ans;
}


// LC545
vector<int> boundaryOfBinaryTree(TreeNode* root)
{
    vector<int> ans;
    list<TreeNode *> left;
    unordered_set<TreeNode *> maybeRightNode, maybeLeftNode;
    // 左边界
    TreeNode * t = root;
    if (t->left != nullptr) {
        while (1) {
            if (t->left) {
                t = t->left;
                left.emplace_back(t);
            } else if (t->right) {
                t = t->right;
                left.emplace_back(t);
                // 此节点可能会占用右节点, 记录
                maybeRightNode.emplace(t);
            } else {
                break;
            }
        }
    }
    // 右边界
    list<TreeNode *> right;
    t = root;
    if (t->right != nullptr) {
        while (1) {
            if (t->right) {
                t = t->right;
                right.emplace_back(t);
            } else if (t->left) {
                t = t->left;
                right.emplace_back(t);
                // 此节点可能会占用左节点, 记录
                maybeLeftNode.emplace(t);
            } else {
                break;
            }
        }
    }
    // 去除左边界与右边界重复的点
    for (auto it : left) {
        // 如果这个节点在右边界中, 删除
        auto Right_iter = find(right.begin(), right.end(), it);
        if (Right_iter != right.end() && maybeLeftNode.count(it)) {
            right.erase(Right_iter);
        }
    }
    for (auto it : right) {
        // 如果这个节点在左边界中, 删除
        auto Left_iter = find(left.begin(), left.end(), it);
        if (Left_iter != left.end() && maybeRightNode.count(it)) {
            left.erase(Left_iter);
        }
    }
    // 将右边界逆序
    reverse(right.begin(), right.end());
    // 叶子节点
    list<TreeNode *> leaf;
    function<void (TreeNode *)> Preorder = [&Preorder, &leaf, &root](TreeNode *node) {
        if (node == nullptr) {
            return;
        }
        if (node != root && node->left == nullptr && node->right == nullptr) {
            leaf.emplace_back(node);
            return;
        }
        Preorder(node->left);
        Preorder(node->right);
    };
    Preorder(root);
    // 去除leaf里面与right和left重复的点
    for (auto it = leaf.begin(); it != leaf.end();) {
        auto Left_iter = find(left.begin(), left.end(), *it);
        auto Right_iter = find(right.begin(), right.end(), *it);
        if (Left_iter != left.end()) {
            leaf.erase(it++);
        } else if (Right_iter != right.end()) {
            leaf.erase(it++);
        } else {
            it++;
        }
    }
    // 组合答案
    ans.emplace_back(root->val);
    for (auto it : left) {
        ans.emplace_back(it->val);
    }
    for (auto it : leaf) {
        ans.emplace_back(it->val);
    }
    for (auto it : right) {
        ans.emplace_back(it->val);
    }
    return ans;
}


// LC2182
string repeatLimitedString(string s, int repeatLimit)
{
    map<char, int, greater<>> m;
    for (auto ch : s) {
        m[ch]++;
    }
    string t;
    char ch;
    while (1) {
        auto it = m.begin();
        if (it == m.end()) {
            goto repeatLimitedStringEND;
        }
        for (; it != m.end();) {
            if (t.empty()) {
                if (it->second <= repeatLimit) {
                    t.append(it->second, it->first);
                    m.erase(it++);
                } else {
                    t.append(repeatLimit, it->first);
                    m[it->first] -= repeatLimit;
                }
                break;
            } else {
                ch = t[t.size() - 1];
                if (ch == it->first) {
                    it++;
                    if (it == m.end()) {
                        goto repeatLimitedStringEND;
                    }
                    continue;
                }
                if (it == m.begin()) {
                    if (it->second <= repeatLimit) {
                        t.append(it->second, it->first);
                        m.erase(it++);
                    } else {
                        t.append(repeatLimit, it->first);
                        m[it->first] -= repeatLimit;
                    }
                } else {
                    t.append(1, it->first);
                    if (it->second == 1) {
                        m.erase(it++);
                    } else {
                        it->second--;
                    }
                }
                break;
            }
        }
    }
repeatLimitedStringEND:
    return t;
}


// LC998
TreeNode* insertIntoMaxTree(TreeNode* root, int val)
{
    vector<int> v;
    // 重建原数组
    function<vector<int> (TreeNode *)> RebuildVector = [&RebuildVector](TreeNode *node) {
        if (node == nullptr) {
            return vector<int>();
        }
        vector<int> v;
        vector<int> left = RebuildVector(node->left);
        vector<int> right = RebuildVector(node->right);
        v.insert(v.end(), left.begin(), left.end());
        v.emplace_back(node->val);
        v.insert(v.end(), right.begin(), right.end());
        return v;
    };
    v = RebuildVector(root);

    v.emplace_back(val);
    // 从数组构建新树
    TreeNode *newroot = nullptr;
    function<TreeNode *(vector<int>&)> BuildTree = [&BuildTree](vector<int>& v) {
        if (v.empty()) {
            return static_cast<TreeNode *>(nullptr);
        }
        vector<int> left;
        vector<int> right;
        int maxVal = *max_element(v.begin(), v.end());
        int i, j;
        int n = v.size();
        for (i = 0; i < n; i++) {
            if (v[i] != maxVal) {
                left.emplace_back(v[i]);
            } else {
                j = i + 1;
                break;
            }
        }
        for (i = j; i < n; i++) {
            right.emplace_back(v[i]);
        }
        TreeNode *node = new TreeNode(maxVal);
        node->left = BuildTree(left);
        node->right = BuildTree(right);
        return node;
    };
    newroot = BuildTree(v);
    return newroot;
}


// LC2979
int mostExpensiveItem(int primeOne, int primeTwo)
{
    int i, j, k;
    int n = primeOne * primeTwo;
    vector<int> dp(n + 1, 0);

    dp[0] = 0;
    i = 0;
    j = 0;
    for (k = 1; k <= n; k++) {
        dp[k] = min(dp[i] + primeOne, dp[j] + primeTwo);
        if (dp[k] == dp[i] + primeOne) {
            i++;
        }
        if (dp[k] == dp[j] + primeTwo) {
            j++;
        }
    }
    for (k = n - 1; k >= 0; k--) {
        if (dp[k] + 1 != dp[k + 1]) {
            return dp[k] + 1;
        }
    }
    return 0;
}


// LC777
bool canTransform(string start, string end)
{
    int i;
    int m = start.size();
    int n = end.size();

    if (m != n) {
        return false;
    }
    string t1, t2;
    vector<int> idxStart, idxEnd;

    i = 0;
    for (auto ch : start) {
        if (ch != 'X') {
            t1 += ch;
            idxStart.emplace_back(i);
        }
        i++;
    }
    i = 0;
    for (auto ch : end) {
        if (ch != 'X') {
            t2 += ch;
            idxEnd.emplace_back(i);
        }
        i++;
    }
    if (t1 != t2) {
        return false;
    }
    for (i = 0; i < idxStart.size(); i++) {
        if (start[idxStart[i]] == 'L' && idxStart[i] < idxEnd[i]) {
            return false;
        }
        if (start[idxStart[i]] == 'R' && idxStart[i] > idxEnd[i]) {
            return false;
        }
    }
    return true;
}


// LC2488
int countSubarrays(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int ans = 0;
    unordered_map<int, int> um;
    vector<int> prefixSum(n, 0);

    auto idx = find(nums.begin(), nums.end(), k) - nums.begin();
    if (nums[0] < k) {
        prefixSum[0] = -1;
        um[-1] = 1;
    } else if (nums[0] > k) {
        prefixSum[0] = 1;
        um[1] = 1;
    } else {
        prefixSum[0] = 0;
        ans++;
    }
    for (i = 1; i < n; i++) {
        if (nums[i] < k) {
            prefixSum[i] = prefixSum[i - 1] - 1;
        } else if (nums[i] > k) {
            prefixSum[i] = prefixSum[i - 1] + 1;
        } else {
            prefixSum[i] = prefixSum[i - 1];
        }
        if (i < idx) {
            um[prefixSum[i]]++;
            continue;
        }
        if (prefixSum[i] == 0 || prefixSum[i] == 1) {
            ans++;
        }
        if (um.count(prefixSum[i] - 1)) {
            ans += um[prefixSum[i] - 1];
        }
        if (um.count(prefixSum[i])) {
            ans += um[prefixSum[i]];
        }
        // um[prefixSum[i]]++; 不再记录, 因为nums[idx]必须包含
    }
    return ans;
}


// LC2054
int maxTwoEvents(vector<vector<int>>& events)
{
    sort(events.begin(), events.end(), [](vector<int>& a, vector<int>& b) {
            if (a[1] != b[1]) {
                return a[1] < b[1];
            }
            return a[0] < b[0];
        }
    );

    int i;
    int n = events.size();
    int ans;
    vector<int> dp(n, 0);

    // pair<int, int> - {开始时间, val} 按val从大到小, 开始时间从小到大排
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        }
        return a.first > b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);
    
    for (i = 0; i < n; i++) {
        pq.push({events[i][0], events[i][2]});
    }
    ans = 0;
    for (i = 0; i < n; i++) {
        dp[i] = events[i][2];
        while (!pq.empty()) {
            auto top = pq.top();
            if (top.first > events[i][1]) {
                dp[i] += top.second;
                break;
            }
            pq.pop();
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC2171
long long minimumRemoval(vector<int>& beans)
{
    sort(beans.begin(), beans.end());

    int i;
    int n = beans.size();
    vector<long long> prefixSum(n, 0);
    unordered_map<int, int> first; // 某个值第一次出现时的下标
    unordered_map<int, int> last; // 某个值最后一次出现时的下标

    prefixSum[0] = beans[0];
    first[beans[0]] = 0;
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + beans[i];
        if (beans[i] != beans[i - 1]) {
            first[beans[i]] = i;
            last[beans[i - 1]] = i - 1;
        }
    }
    last[beans[n - 1]] = n - 1;

    long long ans = LONG_LONG_MAX;
    long long cur, left, right;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            cur = prefixSum[n - 1] - prefixSum[last[beans[i]]] - static_cast<long long>(beans[i]) * (n - 1 - last[beans[i]]);
        } else if (i == n - 1) {
            if (first[beans[i]] == 0) {
                cur = 0;
                continue;
            }
            cur = prefixSum[first[beans[i]] - 1];
        } else {
            if (first[beans[i]] == 0) {
                left = 0;
            } else {
                left = prefixSum[first[beans[i]] - 1];
            }
            if (last[beans[i]] == n - 1) {
                right = 0;
            } else {
                right = prefixSum[n - 1] - prefixSum[last[beans[i]]] - static_cast<long long>(beans[i]) * (n - 1 - last[beans[i]]);
            }
            cur = left + right;
        }
        ans = min(ans, cur);
    }
    return ans;
}


// LC1223
int dieSimulator(int n, vector<int>& rollMax)
{
    int mod = 1e9 + 7;
    // dp[i][j][k] - 第i次投骰子为j且已经连续投k次j的总序列数
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(7, vector<long long>(16, 0)));

    int i, j, k;
    int jCur, jPrev;
    for (i = 1; i <= 6; i++) {
        dp[0][i][1] = 1;
    }
    for (i = 1; i < n; i++) {
        for (jCur = 1; jCur <= 6; jCur++) {
            for (jPrev = 1; jPrev <= 6; jPrev++) {
                for (k = 1; k <= 15; k++) {
                    if (k > i + 1 || (k == rollMax[jPrev - 1] && jCur == jPrev)) {
                        break;
                    }
                    if (jCur == jPrev) {
                        dp[i][jCur][k + 1] = (dp[i][jCur][k + 1] + dp[i - 1][jPrev][k]) % mod;
                    } else {
                        dp[i][jCur][1] = (dp[i][jCur][1] + dp[i - 1][jPrev][k]) % mod;
                    }
                }
            }
        }
    }
    long long ans = 0;
    for (i = 1; i <= 6; i++) {
        for (j = 1; j <= rollMax[i - 1]; j++) {
            ans = (ans + dp[n - 1][i][j]) % mod;
        }
    }
    return ans;
}


// LC1361
bool validateBinaryTreeNodes(int n, vector<int>& leftChild, vector<int>& rightChild)
{
    int i;
    unordered_map<int, int> outDegree, inDegree;
    unordered_map<int, vector<int>> edges;

    for (i = 0; i < n; i++) {
        if (leftChild[i] != -1) {
            if (leftChild[leftChild[i]] == i || rightChild[leftChild[i]] == i) { // 双向边
                return false;
            }
            outDegree[i]++;
            inDegree[leftChild[i]]++;
            edges[i].emplace_back(leftChild[i]);
        }
        if (rightChild[i] != -1) {
            if (leftChild[rightChild[i]] == i || rightChild[rightChild[i]] == i) { // 双向边
                return false;
            }
            outDegree[i]++;
            inDegree[rightChild[i]]++;
            edges[i].emplace_back(rightChild[i]);
        }
    }
    int m = inDegree.size();
    if (n - m != 1) { // 入度为0(根)不止一个或没有根
        return false;
    }
    for (auto it : inDegree) {
        if (it.second > 1) { // 图
            return false;
        }
    }
    for (auto it : outDegree) {
        if (it.second > 2) { // 多个子节点
            return false;
        }
    }
    // 判断环
    vector<int> visited(n, 0);
    function<void (int, vector<int>&, bool&)> Loop = [&edges, &Loop](int cur, vector<int>& visited, bool& find) {
        if (find) {
            return;
        }
        if (visited[cur] == 2) {
            find = true;
            return;
        }
        visited[cur] = 2; // 正在被访问
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                Loop(it, visited, find);
            }
        }
        visited[cur] = 1; // 访问完成
    };
    bool find = false;
    for (i = 0; i < n; i++) {
        if (visited[i] == 0) {
            find = false;
            Loop(i, visited, find);
            if (find) {
                return false;
            }
        }
    }
    return true;
}


// LC1229
vector<int> minAvailableDuration(vector<vector<int>>& slots1, vector<vector<int>>& slots2, int duration)
{
    sort(slots1.begin(), slots1.end());
    sort(slots2.begin(), slots2.end());

    int i, j;
    int start, end;
    int m = slots1.size();
    int n = slots2.size();

    i = j = 0;
    while (i < m && j < n) {
        if (slots2[j][0] > slots1[i][1]) {
            i++;
        } else if (slots1[i][0] > slots2[j][1]) {
            j++;
        } else {
            if (slots2[j][0] <= slots1[i][0]) {
                start = slots1[i][0];
                end = min(slots1[i][1], slots2[j][1]);
            } else if (slots2[j][1] < slots1[i][1]) {
                start = slots2[j][0];
                end = slots2[j][1];
            } else {
                start = slots2[j][0];
                end = slots1[i][1];
            }
            if (end - start >= duration) {
                return {start, start + duration};
            } else {
                if (slots2[j][1] > slots1[i][1]) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }
    return {};
}


// LC3008 LC1392
// KMP
void GenerateNextArr(string& s, vector<int>& next)
{
    int i, j;
    int n = s.size();
    next.resize(n);

    next[0] = 0;
    j = 0;
    for (i = 1; i < n; i++) {
        while (s[i] != s[j]) {
            if (j == 0) {
                break;
            }
            j = next[j - 1];
        }
        if (s[i] == s[j]) {
            next[i] = j + 1;
            j++;
        } else {
            next[i] = 0;
        }
    }
}
vector<int> beautifulIndices(string s, string a, string b, int k)
{
    int i, j;
    vector<int> nextA, nextB;

    GenerateNextArr(a, nextA);
    GenerateNextArr(b, nextB);

    vector<int> idxa, idxb;
    int n = s.size();
    int m = a.size();

    j = 0;
    for (i = 0; i < n; i++) {
        while (s[i] != a[j]) {
            if (j == 0) {
                break;
            }
            j = nextA[j - 1];
        }
        if (s[i] == a[j]) {
            j++;
        }
        if (j == m) {
            idxa.emplace_back(i - j + 1);
        }
    }

    j = 0;
    m = b.size();
    for (i = 0; i < n; i++) {
        while (s[i] != b[j]) {
            if (j == 0) {
                break;
            }
            j = nextB[j - 1];
        }
        if (s[i] == b[j]) {
            j++;
        }
        if (j == m) {
            idxb.emplace_back(i - j + 1);
        }
    }
    vector<int> ans;
    int left, right, mid;
    for (i = 0; i < idxa.size(); i++) {
        left = 0;
        right = idxb.size() - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (idxb[mid] < idxa[i] && abs(idxb[mid] - idxa[i]) > k) {
                left = mid + 1;
            } else if (idxb[mid] > idxa[i] && abs(idxb[mid] - idxa[i]) > k) {
                right = mid - 1;
            } else {
                ans.emplace_back(idxa[i]);
                break;
            }
        }
    }
    return ans;
}


// LC3011
bool canSortArray(vector<int>& nums)
{
    unordered_map<int, int> bitCnt;

    int cnt;
    int n = nums.size();
    int i;
    for (i = 0; i < n; i++) {
        cnt = 0;
        auto t = nums[i];
        while (t) {
            if (t % 2 == 1) {
                cnt++;
            }
            t /= 2;
        }
        bitCnt[nums[i]] = cnt;
    }
    for (i = 1; i < n; i++) {
        while (nums[i] < nums[i - 1]) {
            if (bitCnt[nums[i]] != bitCnt[nums[i - 1]]) {
                return false;
            }
            swap(nums[i], nums[i - 1]);
            i--;
            if (i == 0) {
                break;
            }
        }
    }
    return true;
}


// LC3012
int minimumArrayLength(vector<int>& nums)
{
    map<int, int> cnt;
    for (auto n : nums) {
        cnt[n]++;
    }

    vector<int> v;
    for (auto it : cnt) {
        v.emplace_back(it.first);
    }
    if (cnt[v[0]] == 1) {
        return 1;
    }

    int i;
    int n = v.size();
    for (i = 1; i < n; i++) {
        if (v[i] % v[0] != 0) {
            return 1;
        }
    }
    return (cnt[v[0]] + 1) / 2;
}


// LC1156
int maxRepOpt1(string text)
{
    int i, k;
    int n = text.size();
    int ans, size;
    int start, end;
    bool f = false;
    unordered_set<int> us;
    vector<pair<int, int>> vp;
    for (auto ch : text) {
        us.emplace(ch);
    }
    ans = 0;
    for (auto it : us) {
        vp.clear();
        f = false;
        for (i = 0; i < n; i++) {
            if (text[i] == it) {
                if (f == false) {
                    f = true;
                    start = i;
                }
            } else {
                if (f) {
                    f = false;
                    end = i - 1;
                    vp.push_back({start, end});
                }
            }
        }
        if (f) {
            vp.push_back({start, n - 1});
        }

        size = vp.size();
        if (size == 1) {
            ans = max(ans, vp[0].second - vp[0].first + 1);
        } else {
            ans = max(ans, 1 + vp[0].second - vp[0].first + 1);
        }
        for (k = 1; k < size; k++) {
            if (vp[k - 1].second + 2 == vp[k].first) {
                if (size == 2) {
                    ans = max(ans, 
                    vp[k - 1].second - vp[k - 1].first + 1 + vp[k].second - vp[k].first + 1);
                } else { // 从第3个字符串"借"一个
                    ans = max(ans, 1 +
                    vp[k - 1].second - vp[k - 1].first + 1 + vp[k].second - vp[k].first + 1);
                }
            } else {
                ans = max(ans, 1 + vp[k].second - vp[k].first + 1);
            }
        }
    }
    return ans;
}


// LC2121
vector<long long> getDistances(vector<int>& arr)
{
    int i;
    int n = arr.size();
    unordered_map<int, vector<int>> idx;

    for (i = 0; i < n; i++) {
        idx[arr[i]].emplace_back(i);
    }
    // 计算差值前缀和
    unordered_map<int, vector<long long>> diffPreSum;
    for (auto it : idx) {
        auto v = it.second;
        n = v.size();
        diffPreSum[it.first].emplace_back(0);
        for (i = 1; i < n; i++) {
            diffPreSum[it.first].emplace_back(
                diffPreSum[it.first][diffPreSum[it.first].size() - 1] + idx[it.first][i] - idx[it.first][0]);
        }
    }
    // 计算差值后缀和
    unordered_map<int, vector<long long>> diffSufSum;
    for (auto it : idx) {
        auto v = it.second;
        n = v.size();
        diffSufSum[it.first].emplace_back(0);
        for (i = n - 2; i >= 0; i--) {
            diffSufSum[it.first].emplace_back(
                diffSufSum[it.first][diffSufSum[it.first].size() - 1] + idx[it.first][n - 1] - idx[it.first][i]);
        }
    }

    unordered_map<int, int> um; // 遍历时arr[i]出现的次数
    n = arr.size();
    vector<long long> ans(n);
    long long pre, suf;
    for (i = 0; i < n; i++) {
        if (idx[arr[i]].size() == 1) {
            ans[i] = 0;
        } else {
            pre = diffPreSum[arr[i]][diffPreSum[arr[i]].size() - 1];
            suf = diffSufSum[arr[i]][diffSufSum[arr[i]].size() - 1];
            if (um[arr[i]] == 0) {
                ans[i] = pre;
            } else if (um[arr[i]] == diffPreSum[arr[i]].size() - 1) {
                ans[i] = suf;
            } else {
                pre = static_cast<long long>(um[arr[i]]) * (idx[arr[i]][um[arr[i]]] - idx[arr[i]][0]) - diffPreSum[arr[i]][um[arr[i]] - 1];
                suf = static_cast<long long>(idx[arr[i]].size() - 1 - um[arr[i]]) * 
                    (idx[arr[i]][idx[arr[i]].size() - 1] - idx[arr[i]][um[arr[i]]]) - diffSufSum[arr[i]][idx[arr[i]].size() - 1 - um[arr[i]] - 1];
                ans[i] = pre + suf;
            }
            um[arr[i]]++;
        }
    }
    return ans;
}


// LC3004
int maximumSubtreeSize(vector<vector<int>>& e, vector<int>& colors)
{
    int i;
    int n = colors.size();
    unordered_map<int, vector<int>> edges;

    if (e.size() == 0) {
        return 1;
    }
    for (auto it : e) {
        edges[it[0]].emplace_back(it[1]);
        edges[it[1]].emplace_back(it[0]);
    }

    unordered_map<int, int> colorCnt;
    function<int (unordered_map<int, vector<int>>&, int, int)> find = 
        [&colors, &colorCnt, &find](unordered_map<int, vector<int>>& edges, int cur, int parent) {
        bool flag = true;
        if (edges[cur].size() == 1 && edges[cur][0] == parent) {
            colorCnt[cur] = 1;
            return 1;
        }
        for (auto it : edges[cur]) {
            if (it != parent) {
                colorCnt[it] = find(edges, it, cur);
                if (colors[cur] == colors[it] && flag) {
                    if (colorCnt[it] != 0) {
                        colorCnt[cur] += colorCnt[it];
                    } else {
                        colorCnt[cur] = 0;
                        flag = false;
                    }
                } else {
                    flag = false;
                    colorCnt[cur] = 0;
                }
            }
        }
        if (colorCnt[cur] != 0) {
            colorCnt[cur]++;
        }
        return colorCnt[cur];
    };
    find(edges, 0, -1);

    int ans = 1;
    for (auto it : colorCnt) {
        ans = max(ans, it.second);
    }
    return ans;
}


// LC780
bool reachingPoints(int sx, int sy, int tx, int ty)
{
    if (sx > tx || sy > ty) {
        return false;
    }
    if (sx == tx && sy == ty) {
        return true;
    }
    int tmpX, tmpY;
    while (1) {
        if (tx > ty) {
            tmpX = (tx - sx) % ty + sx;
            if (tmpX == tx) {
                return false;
            }
            tx = tmpX;
        } else if (ty > tx) {
            tmpY = (ty - sy) % tx + sy;
            if (tmpY == ty) {
                return false;
            }
            ty = tmpY;
        } else {
            return false;
        }
        if (sx == tx && sy == ty) {
            return true;
        }
        if (tx < sx || ty < sy) {
            break;
        }
    }
    return false;
}


// LC2438
vector<int> productQueries(int n, vector<vector<int>>& queries)
{
    vector<long long> powers;
    long long i = 1;
    int mod = 1e9 + 7;
    while (i <= n) {
        if ((n & i) == i) {
            powers.emplace_back(i);
        }
        i <<= 1;
    }
    long long t;
    vector<int> ans;
    for (auto q : queries) {
        t = 1;
        for (i = q[0]; i <= q[1]; i++) {
            t = t * powers[i] % mod;
        }
        ans.emplace_back(t);
    }
    return ans;
}


// LC2585
int waysToReachTarget(int target, vector<vector<int>>& types)
{
    int i, j, k;
    int n = types.size();
    int mod = 1e9 + 7;
    vector<vector<long long>> dp(target + 1, vector<long long>(n, 0)); // dp[i][j] - 前j个题目获得i分的方法数

    for (j = 0; j < n; j++) {
        dp[0][j] = 1;
    }
    for (j = 0; j < n; j++) {
        if (j == 0) {
            for (k = 1; k <= types[j][0]; k++) {
                if (k * types[j][1] <= target) {
                    dp[k * types[j][1]][j] = 1;
                } else {
                    break;
                }
            }
            continue;
        }
        for (i = 1; i <= target; i++) {
            for (k = 1; k <= types[j][0]; k++) {
                if (i - k * types[j][1] >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - k * types[j][1]][j - 1]) % mod;
                } else {
                    break;
                }
            }
            dp[i][j] = (dp[i][j] + dp[i][j - 1]) % mod; // 如果k = 0开始循环, 此句不要
        }
    }
    return dp[target][n - 1];
}


// LC1187
int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2)
{
    int i, j, k;
    int idx;
    int n = arr1.size();
    set<int> arr2Set;

    for (auto a : arr2) {
        arr2Set.emplace(a);
    }
    arr2.clear();
    for (auto it : arr2Set) {
        arr2.emplace_back(it);
    }

    int m = arr2.size();
    // dp[i][0] - arr1[i]不替换最小操作数
    // dp[i][j] (j >= 1) - arr1[i]被arr2[j - 1]替换最小操作数
    vector<vector<int>> dp(n, vector<int>(m + 1, 0x3f3f3f3f));

    dp[0][0] = 0;
    for (j = 1; j <= m; j++) {
        dp[0][j] = 1;
    }
    for (i = 1; i < n; i++) {
        // i 不操作 i - 1 也不操作
        if (arr1[i] > arr1[i - 1]) {
            dp[i][0] = min(dp[i][0], dp[i - 1][0]);
        }
        // i 不操作 i - 1 操作
        idx = lower_bound(arr2.begin(), arr2.end(), arr1[i]) - arr2.begin();
        cout << "1:" << idx << endl;
        for (j = 1; j <= idx; j++) {
            dp[i][0] = min(dp[i][0], dp[i - 1][j]);
        }
        // i 操作, i - 1 不操作
        idx = upper_bound(arr2.begin(), arr2.end(), arr1[i - 1]) - arr2.begin();
        cout << "2:" << idx << endl;
        for (j = idx + 1; j <= m; j++) {
            dp[i][j] = min(dp[i][j], dp[i - 1][0] + 1);
        }
        // i - 1 与 i 都操作
        for (j = 2; j <= m; j++) {
            dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1);
        }
    }
    int ans = 0x3f3f3f3f;
    ans = min(ans, *min_element(dp[n - 1].begin(), dp[n - 1].end()));
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC768
int maxChunksToSorted(vector<int>& arr)
{
    map<int, int> arrCnt;

    for (auto a : arr) {
        arrCnt[a]++;
    }
    int curMaxNum, ans;

    curMaxNum = -1;
    ans = 0;
    for (auto num : arr) {
        curMaxNum = max(curMaxNum, num);
        if (arrCnt[num] == 1) {
            arrCnt.erase(num);
        } else {
            arrCnt[num]--;
        }
        if (arrCnt.empty() || curMaxNum <= arrCnt.begin()->first) {
            ans++;
        }
    }
    return ans;
}