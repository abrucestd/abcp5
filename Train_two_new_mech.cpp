#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <windows.h>
#include <cstring>
#include <vector>
#include <map>
#include <unordered_map>
#include <omp.h>
#include <pthread.h>
#include<fstream>
#include <unistd.h>
#ifdef _WIN32
#include<fcntl.h>
#include<io.h>
#endif
// #pragma GCC optimize("Ofast")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,abm,mmx,avx,avx2")
// #pragma GCC optimize("unroll-loops")
#define RAW_LOG(msg) \
    write(2, msg, strlen(msg))
typedef unsigned long long u64_t;
typedef unsigned char u8_t;
using namespace std;
const char PRE='S';
int XPmin=4900,XDmin=5700,XD_XPmin=3500;
const char skillNameMap[][13] = {
	"火球术", "冰冻术", "雷击术", "地裂术", "吸血攻击", "投毒", "连击",
	"会心一击", "瘟疫", "生命之轮", "狂暴术", "魅惑", "加速术", "减速术",
	"诅咒", "治愈魔法", "苏生术", "净化", "铁壁", "蓄力", "聚气",
	"潜行", "血祭", "分身", "幻术", "防御", "守护", "伤害反弹",
	"护身符", "护盾", "反击", "吞噬", "召唤亡灵", "垂死抗争", "隐匿",
	"啧", "啧", "啧", "啧", "啧"};
string skillNameMap_2[35] = {
	"火球", "冰冻", "雷击", "地裂", "吸血", "投毒", "连击",
	"会心", "瘟疫", "命轮", "狂暴", "魅惑", "加速", "减速",
	"诅咒", "治愈", "苏生", "净化", "铁壁", "蓄力", "聚气",
	"潜行", "血祭", "分身", "幻术", "防御", "守护", "反弹",
	"护符", "护盾", "反击", "吞噬", "召灵", "垂死", "隐匿"};
const int N = 256, M = 128, K = 64, skill_cnt = 40, ROLE_FEATURES = 65;
char team[N],tmq[N],fname[N], _tmp[N],suff[N],suf[N],fname2[N];
char charset[N/2];
FILE *fp;
clock_t start;
int charset_len,variable_len;
int name_len=0,suff_len;
int totcnt;
int _collect=0;
int _collect_8V_min,_collect_7V_min,_collect_HL_min,_collect_HP8V_min;
int _output_XP=1;
int _output_log=1;
int _output_speed=1;
map <string,int> Map;
char *dvt(char *p)
{
	DWORD dwNum = MultiByteToWideChar(CP_UTF8, 0, p, -1, NULL, 0);
	char *psText;
	wchar_t *pwText = (wchar_t *)malloc(dwNum * sizeof(wchar_t));
	dwNum = MultiByteToWideChar(CP_UTF8, 0, p, -1, pwText, dwNum);
	dwNum = WideCharToMultiByte(CP_ACP, 0, pwText, -1, NULL, 0, NULL, NULL);
	psText = (char *)malloc(dwNum * sizeof(char));
	dwNum = WideCharToMultiByte(CP_ACP, 0, pwText, -1, psText, dwNum, NULL, NULL);
	free(pwText);
	return psText;
}
char OOO[1024];
void puts_Chinese(string s_cn)
{
	for (int i=0;i<1023;i++) OOO[i]=s_cn[i];
	char *PPP= dvt(OOO);
	fprintf(stderr,"%s",PPP);
}
struct Name
{
	u8_t ual[N],val[N],val_base[N];
	u8_t name_base[M], freq[16], skill[skill_cnt], p, q;
	int q_len,last;
    double cfz;
    double shadowi[9];
    double x[72];
    bool freq14,freq15;
	inline u8_t m()
	{
		q += val[++p];
		swap(val[p], val[q]);
		return val[val[p] + val[q] & 255];
	}
	inline int gen()
	{
		int u = m();
		return (u << 8 | m()) % skill_cnt;
	}
	void load_team(char *_team)
	{
		int t_len = strlen(_team);
		u8_t s;
		for (int i = 0; i < N; i++) val_base[i] = i;
		for (int i = s = 0, j = t_len; i < N; ++i, ++j)
		{
			s += _team[j] + val_base[i];
			swap(val_base[i], val_base[s]);
            if (j==t_len) j=-1;
		}
	}
    
    #define median(x, y, z) (x<y?(x<z?(y<z?y:z):x):(y<z?(x<z?x:z):y))
	void load_shadowname(const char *name)
	{
		memcpy(val, val_base, sizeof val);
		q_len = -1;
		u8_t s;
		int t_len=strlen(name);
		for (int _ = 0; _ < 2; _++)
			for (int i = s = 0, j = t_len; i < N; ++i, ++j)
            {
                s += name[j] + val[i];
                swap(val[i], val[s]);
                if (j==t_len) j=-1;
            }
		q_len = -1;
        for (int i = 0; i < N; i += 8) {
			ual[i + 0] = val[i + 0] * 181 + 160;
			ual[i + 1] = val[i + 1] * 181 + 160;
			ual[i + 2] = val[i + 2] * 181 + 160;
			ual[i + 3] = val[i + 3] * 181 + 160;
			ual[i + 4] = val[i + 4] * 181 + 160;
			ual[i + 5] = val[i + 5] * 181 + 160;
			ual[i + 6] = val[i + 6] * 181 + 160;
			ual[i + 7] = val[i + 7] * 181 + 160;
		}
		for (int i = 0; i < N; i ++)
			if (ual[i] >= 89 && ual[i] < 217)
				name_base[++q_len] = ual[i] & 63;
        memset(shadowi,0,sizeof(shadowi));
        shadowi[0] = 36+median(name_base[10], name_base[11], name_base[12]);
        shadowi[1] = 36+median(name_base[13], name_base[14], name_base[15]);
        shadowi[2] = 36+median(name_base[16], name_base[17], name_base[18]);
        shadowi[3] = 36+median(name_base[19], name_base[20], name_base[21]);
        shadowi[4] = 36+median(name_base[22], name_base[23], name_base[24]);
        shadowi[5] = 36+median(name_base[25], name_base[26], name_base[27]);
        shadowi[6] = 36+median(name_base[28], name_base[29], name_base[30]);
        sort(name_base, name_base + 10);
        shadowi[7] = (154 + name_base[3] + name_base[4] + name_base[5] + name_base[6])/2;
        shadowi[8]=((min(min(name_base[64],name_base[65]),min(name_base[66],name_base[67]))-10)/2+36)*2;
        // cerr<<"step3\n";
	}
    void load_name(const char *name)
    {
        last = -1;
        memcpy(val, val_base, sizeof val);
        q_len = -1;
        u8_t s;
        int t_len=strlen(name);
        for (int _ = 0; _ < 2; _++)
            for (int i = s = 0, j = t_len; i < N; ++i, ++j)
            {
                s += name[j] + val[i];
                swap(val[i], val[s]);
                if (j==t_len) j=-1;
            }
        q_len = -1;
        for (int i = 0; i < N; i += 8) {
            ual[i + 0] = val[i + 0] * 181 + 160;
            ual[i + 1] = val[i + 1] * 181 + 160;
            ual[i + 2] = val[i + 2] * 181 + 160;
            ual[i + 3] = val[i + 3] * 181 + 160;
            ual[i + 4] = val[i + 4] * 181 + 160;
            ual[i + 5] = val[i + 5] * 181 + 160;
            ual[i + 6] = val[i + 6] * 181 + 160;
            ual[i + 7] = val[i + 7] * 181 + 160;
        }
        for (int i = 0; i < N; i ++)
            if (ual[i] >= 89 && ual[i] < 217)
                name_base[++q_len] = ual[i] & 63;
        u8_t *a = name_base + K;
        for (int i = 0; i < skill_cnt; i ++) skill[i] = i;
        memset(freq, 0, sizeof freq);
        p = q = 0;
        for (int s = 0, _ = 0; _ < 2; _ ++)
            for (int i = 0; i < skill_cnt; i ++) {
                s = (s + gen() + skill[i]) % skill_cnt;
                swap(skill[i], skill[s]);
            }
        freq14=freq15=false;
        for (int i = 0, j = 0; i < K; i += 4, j ++) {
            u8_t p = min({a[i], a[i + 1], a[i + 2], a[i + 3]});
            if (p > 10 && skill[j] < 35) {
                if (skill[j] < 25) last = j;
                if (i==56) freq14 = true;
                if (i==60) freq15 = true;
            }
        }
    }
    void get_43()
    {
        sort(name_base, name_base + 10);
        memset(x,0,sizeof(x));
        x[0] = 154 + name_base[3] + name_base[4] + name_base[5] + name_base[6];
        x[1] = 36 + (median(name_base[10], name_base[11], name_base[12]));
        x[2] = 36 + (median(name_base[13], name_base[14], name_base[15]));
        x[3] = 36 + (median(name_base[16], name_base[17], name_base[18]));
        x[4] = 36 + (median(name_base[19], name_base[20], name_base[21]));
        x[5] = 36 + (median(name_base[22], name_base[23], name_base[24]));
        x[6] = 36 + (median(name_base[25], name_base[26], name_base[27]));
        x[7] = 36 + (median(name_base[28], name_base[29], name_base[30]));
        cfz=(((x[1]-x[2]+x[3]+x[5]-x[6])*2+x[4]+x[7])-144)/(min(x[0],300.0))*300;
        u8_t *a = name_base + K;
        
        for (int i = 0, j = 0; i < K; i += 4, j++) {
            u8_t p = min({a[i], a[i + 1], a[i + 2], a[i + 3]});
            if (p > 10 && skill[j] < 35) {
                freq[j] = p - 10;
            } else
                freq[j] = 0;
        }
        int raw_freq[16];
        for (int i = 0; i < 16; i++) raw_freq[i] = freq[i];

        int tail_active_skill = (last >= 0 ? skill[last] : -1);
        int tail_active_raw = (last >= 0 ? raw_freq[last] : 0);
        int tail14_skill = (freq14 && raw_freq[14] > 0 && skill[14] < 35 ? skill[14] : -1);
        int tail15_skill = (freq15 && raw_freq[15] > 0 && skill[15] < 35 ? skill[15] : -1);
        int tail14_raw = (tail14_skill >= 0 ? raw_freq[14] : 0);
        int tail15_raw = (tail15_skill >= 0 ? raw_freq[15] : 0);
        int tail14_bonus = (tail14_skill >= 0 && last != 14) ? min({name_base[60], name_base[61], (u8_t)raw_freq[14]}) : 0;
        int tail15_bonus = (tail15_skill >= 0 && last != 15) ? min({name_base[62], name_base[63], (u8_t)raw_freq[15]}) : 0;

        if (last != -1) freq[last] <<= 1;
        if (freq14 && last != 14) freq[14] += min({name_base[60], name_base[61], freq[14]});
        if (freq15 && last != 15) freq[15] += min({name_base[62], name_base[63], freq[15]});
        double zd=1,kill=1,bd=1;
		double skill_para[25]={1,1,1,0.5,0.75,0.75,1,1,0.75,0.5,1,1,1,0.75,1,0.75,0.2,1,0.75,0.5,0.3,0.75,0.75,0.3,0.75};
		for (int k=0;k<16;k++)
        {
            if (skill[k]<25) {
                x[skill[k]+8]=zd*freq[k];
                zd*=(1-freq[k]*skill_para[skill[k]]/128);
            } else 
            if (skill[k]==31 || skill[k]==32)
            {
                if (freq[k]>=64) freq[k]=64;
                x[skill[k]+8] = kill*freq[k];
                kill*=(1-freq[k]*0.8/128);
            } else x[skill[k]+8]=freq[k];
        }
        if (x[37] <= 60) x[37] = x[37] * x[37] / 60;
        else x[37] = x[37] * 2 - 60;
        if (x[42] > 0) x[43] = 1;
        if (x[37] > 0) x[44] = x[26], x[26]=0;
        if(x[32]>0)
        {
            // cerr<<"ok\n";
            for(int i=0;i<=8;i++)x[45+i]=shadowi[i]/*,cerr<<shadowi[i]<<" \n"[i==8]*/;
        }
        auto decay_affected_skill = [](int sk) {
            return sk == 9 || sk == 15 || sk == 16 || sk == 23 || sk == 24;
        };
        if (x[31] > 0) {
            x[54] = tail_active_skill + 1;
            x[55] = tail_active_raw;
            x[56] = decay_affected_skill(tail_active_skill) ? 1 : 0;
            x[57] = tail14_skill + 1;
            x[58] = tail14_raw;
            x[59] = tail14_bonus;
            x[60] = decay_affected_skill(tail14_skill) ? 1 : 0;
            x[61] = tail15_skill + 1;
            x[62] = tail15_raw;
            x[63] = tail15_bonus;
            x[64] = decay_affected_skill(tail15_skill) ? 1 : 0;
        }
    }
};
int n, jyztxdy, prelen, tp;
long long l, r;
u8_t idx[16];
Name name_initial;
char s_tmp[256];
static unordered_map<string, Name> build_cache;

static inline bool is_trim_char(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == 0 || c == '+';
}

static bool normalize_range(const char *s, int &l, int &r)
{
    while (l <= r && is_trim_char(s[l])) l++;
    while (l <= r && is_trim_char(s[r])) r--;
    return l <= r;
}

Name build(char *s,int l,int r)
{
    if (l < 0 || r < 0 || l > r || !normalize_range(s, l, r)) {
        RAW_LOG("[BUILD PARAM ERROR]\n");
        return name_initial;
    }
    // cerr<<l<<' '<<r<<' '<<s<<'\n';
    Name x,y;int p=-1;
    for (int i=l;i<=r;i++) if (s[i]=='@') p=i;
    if (p <= l || p >= r) {
        RAW_LOG("[BUILD FORMAT ERROR]\n");
        return name_initial;
    }

    string key(s + l, r - l + 1);
    auto it = build_cache.find(key);
    if (it != build_cache.end()) return it->second;

    // cerr<<l<<' '<<r<<' '<<p<<'\n';
    for (int i=p+1;i<=r;i++) s_tmp[i-p-1]=s[i];s_tmp[r-p]=0;

    x.load_team(s_tmp);
    memcpy(y.val_base,x.val_base,sizeof(x.val_base));
    for (int i=l;i<p;i++) s_tmp[i-l]=s[i];s_tmp[p-l]=0;
    // cerr<<"ok1_1\n";
    x.load_name(s_tmp);
// cerr<<"ok1_2\n";
    s_tmp[p-l]='?';s_tmp[p-l+1]='s';s_tmp[p-l+2]='h';s_tmp[p-l+3]='a';s_tmp[p-l+4]='d';s_tmp[p-l+5]='o';s_tmp[p-l+6]='w';s_tmp[p-l+7]=0;
    y.load_shadowname(s_tmp);
    // for(int i=0;i<=8;i++)cerr<<y.shadowi[i]<<' ';
    // system("pause");
// cerr<<"ok1_3\n";
    memcpy(x.shadowi,y.shadowi,sizeof(x.shadowi));
    build_cache.emplace(key, x);
    return x;
}
void cpy(Name &x,Name &y)
{
    memcpy(x.shadowi,y.shadowi,sizeof(x.shadowi));
    x.last=y.last;x.freq14=y.freq14;x.freq15=y.freq15;
    memcpy(x.name_base,y.name_base,sizeof(x.name_base));
    memcpy(x.skill,y.skill,sizeof(x.skill));
}
void read(char *p)
{
    int tot=0;
	while ((*p = getchar()) == '\n') 
    {tot++;if (tot==1000) exit(0);}
    tot=0;
	while ((*++p = getchar()) != '\n')
    {tot++;if (tot==1000) exit(0);}
	*p = 0;
}

double xp_array[5005],xp_x[100],score,scoreQD;
int prop[8];
char NAME_ALL[N*2];
unsigned int NAME_GUARD = 0xDEADBEEF;
int FREQ[N];
void cvt_name()
{
	int len = MultiByteToWideChar(CP_ACP, 0, NAME_ALL, -1, NULL, 0);
	wchar_t *wstr = new wchar_t[len + 1];
	memset(wstr, 0, len + 1);
	MultiByteToWideChar(CP_ACP, 0, NAME_ALL, -1, wstr, len);
	len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
	memset(NAME_ALL, 0, len + 1);
	WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NAME_ALL, len, NULL, NULL);
	if (wstr) delete[] wstr;
}
Name name;
int tot_cnt=0;
const int MM=4363;
double A[MM+5][MM+5];
double model[MM+5];
const double eps=1e-15;
int shadowcnt=0;
bool flag[5000];
static string outbuf;
static string indexbuf;
static FILE *index_fp = nullptr;
static long long output_row = 0;

static inline void append_double(string &out, double v)
{
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%.6g", v);
    out.append(buf, len);
}

static inline void append_ll(string &out, long long v)
{
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%lld", v);
    out.append(buf, len);
}

static inline void append_tsv_escaped(string &out, const string &s)
{
    for (char ch : s) {
        if (ch == '\\') out.append("\\\\");
        else if (ch == '\t') out.append("\\t");
        else if (ch == '\r') out.append("\\r");
        else if (ch == '\n') out.append("\\n");
        else out.push_back(ch);
    }
}

static inline void flush_indexbuf()
{
    if (index_fp != nullptr && !indexbuf.empty()) {
        fwrite(indexbuf.data(), 1, indexbuf.size(), index_fp);
        indexbuf.clear();
    }
}

static inline void append_index_row(
    const string &path,
    int source_record,
    int source_line,
    double raw_score,
    const char *name_all,
    int ll,
    int rr)
{
    if (index_fp == nullptr) return;

    append_ll(indexbuf, output_row);
    indexbuf.push_back('\t');
    append_tsv_escaped(indexbuf, path);
    indexbuf.push_back('\t');
    append_ll(indexbuf, source_record);
    indexbuf.push_back('\t');
    append_ll(indexbuf, source_line);
    indexbuf.push_back('\t');
    append_double(indexbuf, raw_score);
    indexbuf.push_back('\t');
    append_tsv_escaped(indexbuf, string(name_all + ll, rr - ll + 1));
    indexbuf.push_back('\n');

    if (indexbuf.size() >= (1u << 20)) flush_indexbuf();
}

static inline bool split_duo_line(const string &line, int &ll, int &p, int &rr)
{
    if (line.empty() || line.size() >= N * 2) return false;
    memset(NAME_ALL, 0, sizeof(NAME_ALL));
    memcpy(NAME_ALL, line.data(), line.size());

    ll = 0;
    rr = (int)line.size() - 1;
    if (!normalize_range(NAME_ALL, ll, rr)) return false;

    p = -1;
    for (int i = ll; i <= rr; i++) {
        if (NAME_ALL[i] == '+') p = i;
    }
    if (p <= ll || p >= rr) return false;

    int left_l = ll, left_r = p - 1;
    int right_l = p + 1, right_r = rr;
    if (!normalize_range(NAME_ALL, left_l, left_r)) return false;
    if (!normalize_range(NAME_ALL, right_l, right_r)) return false;
    bool left_at = false, right_at = false;
    for (int i = left_l; i <= left_r; i++) if (NAME_ALL[i] == '@') left_at = true;
    for (int i = right_l; i <= right_r; i++) if (NAME_ALL[i] == '@') right_at = true;
    return left_at && right_at;
}

string clan_key(const char *s, int l, int r)
{
    while (l <= r && (s[l] == ' ' || s[l] == 0 || s[l] == '+')) l++;
    while (l <= r && (s[r] == ' ' || s[r] == 0 || s[r] == '+')) r--;
    int at = -1;
    for (int i = l; i <= r; i++) {
        if (s[i] == '@') at = i;
    }
    int begin = (at == -1) ? l : at + 1;
    int end = r;
    while (begin <= end && s[begin] == ' ') begin++;
    while (begin <= end && s[end] == ' ') end--;
    if (begin > end) return "";
    return string(s + begin, s + end + 1);
}

inline bool load_NAME_ALL_checked_raw(const string &s) {
    constexpr int MAXL = N * 2;

    RAW_LOG("[ENTER load_NAME_ALL_checked]\n");

    int len = (int)s.size();
    if (len >= MAXL) {
        RAW_LOG("[OOB] NAME_ALL too long\n");
        return false;
    }

    memcpy(NAME_ALL, s.data(), len);
    NAME_ALL[len] = '\0';

    RAW_LOG("[LEAVE load_NAME_ALL_checked]\n");
    return true;
}


void solve(string path)
{
    cerr<<"go"<<' '<<path<<'\n';
    ifstream input(path);
    if (!(input >> n)) {
    cerr << "read n failed" << endl;
    return;
}
    cerr<<path<<' '<<input.is_open()<<endl;
	cerr<<"n="<<n<<endl;
    Name x,y;
	for (int i=0;i<35;i++) Map[skillNameMap_2[i]]=i;
    for (int __=1;__<=n;__++)
    {
        
        if (__%10000==0) cerr<<"i="<<__<<"\n";
		double real_score;
		if (!(input>>real_score)) {
            cerr << "[WARN] " << path << " ended early at record " << __
                 << " / declared " << n << '\n';
            break;
        }
        double raw_score = real_score;
		real_score*=100;
        string s;
        static long long line_idx = 0;
        line_idx++;
        getline(input, s);

		int ll=0,rr=0,p=-1;
        if (!split_duo_line(s, ll, p, rr)) {
            cerr << "[WARN] skipped malformed record " << __ << " in " << path << '\n';
            continue;
        }
        // cerr<<"before build.\n";
//         if (p == -1) {
//     RAW_LOG("[FORMAT ERROR] no '+' in NAME_ALL\n");
//     RAW_LOG(NAME_ALL);
//     RAW_LOG("\n");
//     continue;   // 或 return;
// }
        bool same_clan = clan_key(NAME_ALL, ll, p - 1) == clan_key(NAME_ALL, p + 1, rr);
        x=build(NAME_ALL,ll,p-1),y=build(NAME_ALL,p+1,rr);
        // cerr<<"ok2\n";
        //cerr<<"build done.\n";
        Name X,Y;
        cpy(X,x);cpy(Y,y);
        if (same_clan) {
            for (int k = 7; k < M; k++)
                if (y.name_base[k - 1] == x.name_base[k])
                    X.name_base[k] = max(X.name_base[k], y.name_base[k]);
            for (int k = 7; k < M; k++)
                if (x.name_base[k - 1] == y.name_base[k])
                    Y.name_base[k] = max(Y.name_base[k], x.name_base[k]);
        }
        X.get_43();Y.get_43();
        if(X.cfz>Y.cfz)swap(X,Y);
        // cerr<<"ok3\n";
        output_row++;
        append_index_row(path, __, __ + 1, raw_score, NAME_ALL, ll, rr);
        append_double(outbuf, real_score);
        outbuf.push_back(',');
        for(int i=0;i<ROLE_FEATURES;i++) {
            append_double(outbuf, X.x[i]);
            outbuf.push_back(',');
        }
        for(int i=0;i<ROLE_FEATURES;i++) {
            append_double(outbuf, Y.x[i]);
            outbuf.push_back(i==ROLE_FEATURES - 1 ? '\n' : ',');
        }
        if (outbuf.size() >= (1u << 20)) {
            fwrite(outbuf.data(), 1, outbuf.size(), stdout);
            outbuf.clear();
        }
        if (NAME_GUARD != 0xDEADBEEF) {
            cerr << "[NAME_ALL CORRUPTED] "
             << "line=" << line_idx
             << " guard=" << hex << NAME_GUARD << dec << '\n';
            exit(1);
        }
    }
    if (!outbuf.empty()) {
        fwrite(outbuf.data(), 1, outbuf.size(), stdout);
        outbuf.clear();
    }
    flush_indexbuf();
    input.close();
}
int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    // _setmode(_fileno(stdout),_O_U8TEXT);
    // _setmode(_fileno(stdin),_O_U8TEXT);
    #endif
    setvbuf(stderr, nullptr, _IONBF, 0);
    freopen("data_mech.csv","w",stdout);
    index_fp = fopen("data_mech.index.tsv", "wb");
    if (index_fp != nullptr) {
        indexbuf.append("csv_row\tsource_file\tsource_record\tsource_line\traw_score\tduo\n");
        cerr<<"[INFO] index file opened for writing\n";
    } else {
        cerr << "[WARN] cannot open data3_mech.index.tsv for write\n";
    }
    solve("lt4500.txt");
    solve("ge4500.txt");
    flush_indexbuf();
    if (index_fp != nullptr) {
        fclose(index_fp);
        index_fp = nullptr;
    }
}
/*
g++ Train_two.cpp -o Train_two.exe -Ofast -funroll-loops -march=native -fopenmp
./Train_two.exe

*/
