#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct Row
{
    double score = 0.0;
    string duo;
    string original;
};

static string trim_copy(const string &s)
{
    size_t l = 0, r = s.size();
    while (l < r && isspace((unsigned char)s[l])) l++;
    while (r > l && isspace((unsigned char)s[r - 1])) r--;
    return s.substr(l, r - l);
}

static bool parse_scored_line(const string &line, Row &row)
{
    string s = trim_copy(line);
    if (s.empty()) return false;

    size_t pos = 0;
    while (pos < s.size() && !isspace((unsigned char)s[pos])) pos++;
    if (pos == 0 || pos >= s.size()) return false;

    string score_token = s.substr(0, pos);
    char *end = nullptr;
    double score = strtod(score_token.c_str(), &end);
    if (end == score_token.c_str() || *end != 0) return false;

    string duo = trim_copy(s.substr(pos));
    if (duo.empty()) return false;

    row.score = score;
    row.duo = duo;
    row.original = s;
    return true;
}

int main(int argc, char **argv)
{
    string input_path = argc >= 2 ? argv[1] : "result.txt";
    string sorted_path = input_path;
    string without_score_path = argc >= 3 ? argv[2] : "result_without_score.txt";
    if (argc >= 4) {
        cerr << "Usage: result_process4.exe [input_score_file] [without_score_file]\n";
        cerr << "The sorted scored result overwrites input_score_file.\n";
        return 1;
    }

    ifstream fin(input_path, ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "[ERROR] cannot open input: " << input_path << '\n';
        return 1;
    }

    vector<Row> rows;
    rows.reserve(1 << 20);
    string line;
    long long read_count = 0, skipped = 0;
    while (getline(fin, line)) {
        read_count++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        Row row;
        if (parse_scored_line(line, row)) {
            rows.push_back(std::move(row));
        } else {
            skipped++;
        }
    }
    fin.close();

    sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
        if (a.score != b.score) return a.score > b.score;
        return a.duo < b.duo;
    });

    ofstream fout(sorted_path, ios::out | ios::binary);
    if (!fout.is_open()) {
        cerr << "[ERROR] cannot open sorted output: " << sorted_path << '\n';
        return 1;
    }

    ofstream fout_without(without_score_path, ios::out | ios::binary);
    if (!fout_without.is_open()) {
        cerr << "[ERROR] cannot open without-score output: " << without_score_path << '\n';
        return 1;
    }

    string outbuf;
    string without_buf;
    outbuf.reserve(1 << 20);
    without_buf.reserve(1 << 20);
    for (const Row &row : rows) {
        outbuf += row.original;
        outbuf.push_back('\n');
        without_buf += row.duo;
        without_buf.push_back('\n');
        if (outbuf.size() >= (1u << 20) || without_buf.size() >= (1u << 20)) {
            fout.write(outbuf.data(), (streamsize)outbuf.size());
            fout_without.write(without_buf.data(), (streamsize)without_buf.size());
            outbuf.clear();
            without_buf.clear();
        }
    }
    if (!outbuf.empty()) fout.write(outbuf.data(), (streamsize)outbuf.size());
    if (!without_buf.empty()) fout_without.write(without_buf.data(), (streamsize)without_buf.size());

    cerr << "[INFO] read=" << read_count
         << " sorted=" << rows.size()
         << " skipped=" << skipped
         << " sorted_output(overwrite)=" << sorted_path
         << " without_score_output=" << without_score_path
         << '\n';
    return 0;
}
