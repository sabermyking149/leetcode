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

#include "pub.h"

using namespace std;

/* int gcd(int a, int b)
{
    if(b == 0) {
        return a;
    }
    return gcd(b, a % b);
} */
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


// 有问题
void DFSCountNode(unordered_map<int, unordered_set<int>>& edges, int curNode, vector<bool>& visited, int& nodeCnt, vector<int>& nums, vector<int>& curNodes, int k)
{
    if (curNodes.size() > 1) {
        for (int i = 0; i < curNodes.size(); i++) {
            if (abs(curNodes[i] - nums[curNode]) == k) {
                return;
            }
        }
    }
    curNodes.emplace_back(nums[curNode]);
    visited[curNode] = true;
    nodeCnt++;
    for (auto it : edges[curNode]) {
        if (visited[it] == false) {
            DFSCountNode(edges, it, visited, nodeCnt, nums, curNodes, k);
        }
    }
}
int beautifulSubsets(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();
    int ans = 0;
    unordered_map<int, unordered_set<int>> edges;
    vector<bool> visited(n, false);
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (abs(nums[i] - nums[j]) != k) {
                edges[i].emplace(j);
                edges[j].emplace(i);
            }
        }
    }
    int nodeCnt = 0;
    vector<int> curNodes;
    for (i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        nodeCnt = 0;
        curNodes.clear();
        DFSCountNode(edges, i, visited, nodeCnt, nums, curNodes, k);
        if (nodeCnt > 0) {
            ans += pow(2, nodeCnt) - 1;
        }
    }
    return ans;
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