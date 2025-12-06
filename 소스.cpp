// main.cpp
// Requires: GLEW, FreeGLUT, GLM
// Compile example (MSVC): cl /EHsc main.cpp /I"path\to\glew\include" /I"path\to\freeglut\include" /I"path\to\glm" /link glew32.lib freeglut.lib opengl32.lib
//어디까지 했더라

#define _USE_MATH_DEFINES

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <unordered_map>
#include <math.h>
#include <cmath>

#define LOSE_SPEED_VAR 0.9
#define SKYBOX_EDGE_LEN_HALF 100
#define ASTEROID_SIZE_BOTTOM_CAP 6
#define ASTEROID_SIZE_DEFAULT_SMALL 6
#define ASTEROID_SIZE_DEFAULT_MEDIUM 12
#define ASTEROID_SIZE_DEFAULT_LARGE 24
#define ASTEROID_SPEED_VAR 0.2//   /this->size

#define EXPECTED_FRAME_PER_SEC 50

#define MAIN_WEAPON_COOLDOWN 0.2
void srand(unsigned int seed);

// ------------ simple shader loader ------------
static std::string loadFile(const char* path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Failed to open " << path << "\n"; return ""; }
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(p, 1024, nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    return p;
}

// ------------ Octahedron generation (per-face duplicated vertices, per-face normal) ------------
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

static std::vector<Vertex> make_octahedron(float s = 1.0f) {
    // 6 unique vertices of a regular octahedron centered at origin:
    // +/- x, +/- y, +/- z (normalized & scaled)
    const glm::vec3 vpx = glm::vec3(s, 0, 0);
    const glm::vec3 vnx = glm::vec3(-s, 0, 0);
    const glm::vec3 vpy = glm::vec3(0, s, 0);
    const glm::vec3 vny = glm::vec3(0, -s, 0);
    const glm::vec3 vpz = glm::vec3(0, 0, s);
    const glm::vec3 vnz = glm::vec3(0, 0, -s);

    // faces (triangles) defined by triplets of these
    std::vector<glm::vec3> P = { vpx, vnx, vpy, vny, vpz, vnz };
    // we'll list faces by positions directly:
    std::vector<std::array<glm::vec3, 3>> faces = {
        // upper pyramid around +z
        { vpx, vpz, vpy },
        { vpy, vpz, vnx },
        { vnx, vpz, vny },
        { vny, vpz, vpx },
        // lower pyramid around -z
        { vpy, vnz, vpx },
        { vpx, vnz, vny },
        { vny, vnz, vnx },
        { vnx, vnz, vpy }
    };

    std::vector<Vertex> verts;
    verts.reserve(faces.size() * 3);
    for (auto& f : faces) {
        glm::vec3 e1 = f[1] - f[0];
        glm::vec3 e2 = f[2] - f[0];
        glm::vec3 n = glm::normalize(glm::cross(e1, e2)); // face normal
        // ensure consistent winding (counter-clockwise when looking from outside)
        for (int i = 0; i < 3; i++) {
            verts.push_back({ f[i], n });
        }
    }
    return verts;
}

static std::vector<Vertex> make_player_char(float s = 1.0f) {
    const glm::vec3 vpx = glm::vec3(0.6*s, 0, -1.2*s);
    const glm::vec3 vnx = glm::vec3(-0.6*s, 0, -1.2*s);
    const glm::vec3 vpy = glm::vec3(0, 0.5*s, 0);
    const glm::vec3 vny = glm::vec3(0, -0.5*s, 0);
    const glm::vec3 vpz = glm::vec3(0, 0, -0.6*s);
    const glm::vec3 vnz = glm::vec3(0, 0, 1.2*s);

    // faces (triangles) defined by triplets of these
    std::vector<glm::vec3> P = { vpx, vnx, vpy, vny, vpz, vnz };
    // we'll list faces by positions directly:
    std::vector<std::array<glm::vec3, 3>> faces = {
        // upper pyramid around +z
        { vpx, vpz, vpy },
        { vpy, vpz, vnx },
        { vnx, vpz, vny },
        { vny, vpz, vpx },
        // lower pyramid around -z
        { vpy, vnz, vpx },
        { vpx, vnz, vny },
        { vny, vnz, vnx },
        { vnx, vnz, vpy }
    };

    std::vector<Vertex> verts;
    verts.reserve(faces.size() * 3);
    for (auto& f : faces) {
        glm::vec3 e1 = f[1] - f[0];
        glm::vec3 e2 = f[2] - f[0];
        glm::vec3 n = glm::normalize(glm::cross(e1, e2)); // face normal
        // ensure consistent winding (counter-clockwise when looking from outside)
        for (int i = 0; i < 3; i++) {
            verts.push_back({ f[i], n });
        }
    }
    return verts;
}

// ------------------ ICOSPHERE (flat shading) --------------------

struct HashVec {
    size_t operator()(const glm::ivec3& v) const noexcept {
        // simple hash for edge midpoint cache
        return ((size_t)v.x << 20) ^ ((size_t)v.y << 10) ^ (size_t)v.z;
    }
};

// returns vector<Vertex> (position + face-normal) for flat shading
static std::vector<Vertex> make_icosphere(int subdivisions)
{
    // ----- 1. Define base icosahedron -----
    const float t = (1.0f + sqrtf(5.0f)) * 0.5f;

    std::vector<glm::vec3> baseVerts = {
        {-1,  t,  0}, {1,  t,  0}, {-1, -t,  0}, {1, -t,  0},
        {0, -1,  t}, {0,  1,  t}, {0, -1, -t}, {0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
    };

    for (auto& v : baseVerts) v = glm::normalize(v);

    std::vector<glm::ivec3> faces = {
        {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
        {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
        {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
        {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1}
    };

    // ----- 2. Subdivide -----
    auto midpoint = [&](int a, int b, auto& cache, auto& verts) {
        glm::ivec3 key(a, b, (a + b));
        if (cache.count(key)) return cache[key];

        glm::vec3 mid = glm::normalize((verts[a] + verts[b]) * 0.5f);
        verts.push_back(mid);
        int idx = verts.size() - 1;
        cache[key] = idx;
        return idx;
        };

    for (int s = 0; s < subdivisions; s++)
    {
        std::unordered_map<glm::ivec3, int, HashVec> cache;
        std::vector<glm::ivec3> newFaces;

        for (auto& f : faces) {
            int a = f.x;
            int b = f.y;
            int c = f.z;

            int ab = midpoint(a, b, cache, baseVerts);
            int bc = midpoint(b, c, cache, baseVerts);
            int ca = midpoint(c, a, cache, baseVerts);

            newFaces.push_back({ a, ab, ca });
            newFaces.push_back({ b, bc, ab });
            newFaces.push_back({ c, ca, bc });
            newFaces.push_back({ ab, bc, ca });
        }

        faces = std::move(newFaces);
    }

    // ----- 3. Flat shading vertices -----
    std::vector<Vertex> out;
    out.reserve(faces.size() * 3);

    for (auto& f : faces) {
        glm::vec3 v0 = baseVerts[f.x];
        glm::vec3 v1 = baseVerts[f.y];
        glm::vec3 v2 = baseVerts[f.z];

        glm::vec3 N = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        out.push_back({ v0, N });
        out.push_back({ v1, N });
        out.push_back({ v2, N });
    }
    return out;
}

// ---some funcs for calcul---
inline double wrap_delta(double d) {
    return d - std::round(d / (SKYBOX_EDGE_LEN_HALF*2)) * (SKYBOX_EDGE_LEN_HALF*2);
}

bool collided_displacement(
    double sx, double sy, double sz,  // 변위
    double x, double y, double z,   // 이동 후 좌표
    double r,                         // 내 물체 반지름
    double ox, double oy, double oz,  // 상대 좌표
    double r2                        // 상대 반지름
) {
    // 이동 전 좌표 = 끝 - 변위
    double x1 = x - sx;
    double y1 = y - sy;
    double z1 = z - sz;

    // 시작점 P0
    double P0[3] = { x1, y1, z1 };

    // 이동 벡터 v = 변위 (단, 토러스에서 보정)
    double v[3] = {
        wrap_delta(sx),
        wrap_delta(sy),
        wrap_delta(sz)
    };

    double vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

    // 상대 물체 중심을 P0 기준으로 가장 가까운 이미지로 이동
    double dQ[3] = {
        wrap_delta(ox - x1),
        wrap_delta(oy - y1),
        wrap_delta(oz - z1)
    };
    double Qp[3] = {
        x1 + dQ[0],
        y1 + dQ[1],
        z1 + dQ[2]
    };

    // 이동량이 0이면 점-점 거리만 검사
    if (vv == 0.0) {
        double dx = Qp[0] - P0[0];
        double dy = Qp[1] - P0[1];
        double dz = Qp[2] - P0[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        return dist <= (r + r2);
    }

    // (Q' - P0)·v
    double qp[3] = {
        Qp[0] - P0[0],
        Qp[1] - P0[1],
        Qp[2] - P0[2]
    };

    double dot = qp[0] * v[0] + qp[1] * v[1] + qp[2] * v[2];

    // t* = clamp(...)
    double t = dot / vv;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;

    // 선분에서 가장 가까운 점 C
    double C[3] = {
        P0[0] + v[0] * t,
        P0[1] + v[1] * t,
        P0[2] + v[2] * t
    };

    // 최종 거리
    double dx = Qp[0] - C[0];
    double dy = Qp[1] - C[1];
    double dz = Qp[2] - C[2];
    double dmin = std::sqrt(dx * dx + dy * dy + dz * dz);

    return dmin <= (r + r2);
}


glm::vec3 facetovec(glm::vec3 face)
{
    glm::vec4 facetest = glm::vec4(0.0, 0.0, 1.0, 1.0);
    glm::mat4 facetest_mat = glm::mat4(1.0);
    //facetest_mat = glm::translate(facetest_mat, glm::vec3(0.0, 0.0, 1.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.x), glm::vec3(-1.0, 0.0, 0.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.z), glm::vec3(0.0, 0.0, -1.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.y), glm::vec3(0.0, -1.0, 0.0));
    facetest = facetest * facetest_mat;
    glm::vec3 facetest_3vec = glm::vec3(facetest.x, facetest.y, facetest.z);
    return glm::normalize(facetest_3vec);
}
glm::vec3 facetotop(glm::vec3 face)
{
    glm::vec4 facetest = glm::vec4(0.0, 1.0, 0.0, 1.0);
    glm::mat4 facetest_mat = glm::mat4(1.0);
    //facetest_mat = glm::translate(facetest_mat, glm::vec3(0.0, 0.0, 1.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.x), glm::vec3(-1.0, 0.0, 0.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.z), glm::vec3(0.0, 0.0, -1.0));
    facetest_mat = glm::rotate(facetest_mat, glm::radians(face.y), glm::vec3(0.0, -1.0, 0.0));
    facetest = facetest * facetest_mat;
    glm::vec3 facetest_3vec = glm::vec3(facetest.x, facetest.y, facetest.z);
    return glm::normalize(facetest_3vec);
}

glm::vec3 getfowardvec(glm::mat4 dir)
{

    glm::mat4 mat_for_v = glm::scale(dir, glm::vec3(-1.0, -1.0, 1.0));
    glm::vec4 result_v = mat_for_v * glm::vec4(0.0, 0.0, 1.0, 1.0);
    result_v = glm::normalize(result_v);

    return glm::vec3(result_v.x, result_v.y, result_v.z);//glm::normalize(result_v)
}

bool collide_bool(glm::vec3 pos1,glm::vec3 pos2,float r1,float r2)
{
    glm::vec3 delta_pos = pos2 - pos1;
    if (delta_pos.x < 0) delta_pos.x *= -1;
    if (delta_pos.y < 0) delta_pos.y *= -1;
    if (delta_pos.z < 0) delta_pos.z *= -1;
    if (delta_pos.x > SKYBOX_EDGE_LEN_HALF) delta_pos.x = SKYBOX_EDGE_LEN_HALF * 2 - delta_pos.x;
    if (delta_pos.y > SKYBOX_EDGE_LEN_HALF) delta_pos.y = SKYBOX_EDGE_LEN_HALF * 2 - delta_pos.y;
    if (delta_pos.z > SKYBOX_EDGE_LEN_HALF) delta_pos.z = SKYBOX_EDGE_LEN_HALF * 2 - delta_pos.z;
    return pow(delta_pos.x, 2) + pow(delta_pos.y, 2) + pow(delta_pos.z, 2) <= pow(r1 + r2, 2);
}

glm::vec3 altposxyz(glm::vec3 pos, glm::vec3 viewpos)
{
    glm::vec3 altpos = pos;
    if (pos.x > viewpos.x) altpos.x = pos.x - 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.x = pos.x + 2 * SKYBOX_EDGE_LEN_HALF;
    if (pos.y > viewpos.y) altpos.y = pos.y - 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.y = pos.y + 2 * SKYBOX_EDGE_LEN_HALF;
    if (pos.z > viewpos.z) altpos.z = pos.z - 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.z = pos.z + 2 * SKYBOX_EDGE_LEN_HALF;
    return altpos;
}

glm::vec3 altposxyzm(glm::vec3 pos, glm::vec3 viewpos)
{
    glm::vec3 altpos = pos;
    if (pos.x > viewpos.x) altpos.x = pos.x - 4 * SKYBOX_EDGE_LEN_HALF;
    else altpos.x = pos.x - 2 * SKYBOX_EDGE_LEN_HALF;
    if (pos.y > viewpos.y) altpos.y = pos.y - 4 * SKYBOX_EDGE_LEN_HALF;
    else altpos.y = pos.y - 2 * SKYBOX_EDGE_LEN_HALF;
    if (pos.z > viewpos.z) altpos.z = pos.z - 4 * SKYBOX_EDGE_LEN_HALF;
    else altpos.z = pos.z - 2 * SKYBOX_EDGE_LEN_HALF;
    return altpos;
}

glm::vec3 altposxyzp(glm::vec3 pos, glm::vec3 viewpos)
{
    glm::vec3 altpos = pos;
    if (pos.x > viewpos.x) altpos.x = pos.x + 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.x = pos.x + 4 * SKYBOX_EDGE_LEN_HALF;
    if (pos.y > viewpos.y) altpos.y = pos.y + 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.y = pos.y + 4 * SKYBOX_EDGE_LEN_HALF;
    if (pos.z > viewpos.z) altpos.z = pos.z + 2 * SKYBOX_EDGE_LEN_HALF;
    else altpos.z = pos.z + 4 * SKYBOX_EDGE_LEN_HALF;
    return altpos;
}

glm::vec3 posinbox(glm::vec3 pos)
{
    glm::vec3 newpos = pos;
    if (newpos.x > SKYBOX_EDGE_LEN_HALF) newpos.x -= SKYBOX_EDGE_LEN_HALF * 2;
    else if (newpos.x < -SKYBOX_EDGE_LEN_HALF) newpos.x += SKYBOX_EDGE_LEN_HALF * 2;
    if (newpos.y > SKYBOX_EDGE_LEN_HALF) newpos.y -= SKYBOX_EDGE_LEN_HALF * 2;
    else if (newpos.y < -SKYBOX_EDGE_LEN_HALF) newpos.y += SKYBOX_EDGE_LEN_HALF * 2;
    if (newpos.y > SKYBOX_EDGE_LEN_HALF) newpos.z -= SKYBOX_EDGE_LEN_HALF * 2;
    else if (newpos.z < -SKYBOX_EDGE_LEN_HALF) newpos.z += SKYBOX_EDGE_LEN_HALF * 2;
    return newpos;
}
// ------------ Object Classes ------------
class Object {
public:
    Object(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer)
    {
        this->pos = pos;
        this->vel = vel;
        this->rot = glm::mat4(1.0f);
        this->size = this->hitboxr = size;
        this->hp = hp;
        this->timer = timer;
    }

    Object()
    {
        this->pos = this->vel = glm::vec3(0, 0, 0);
        this->rot = glm::mat4(1.0f);
        this->size = 1;
        this->hitboxr = 1;
        this->hp = 1;
        this->timer = 0;
    }

    glm::vec3 getpos()
    {
        return pos;
    }
    glm::vec3 getvel()
    {
        return vel;
    }
    glm::mat4 getfacing()
    {
        return rot;
    }
    float gethp()
    {
        return hp;
    }
    float getsize()
    {
        return size;
    }
    void changehp(float setvalue)
    {
        hp = setvalue;
    }
    void changepos(glm::vec3 setvalue)
    {
        pos = setvalue;
    }
    void changevel(glm::vec3 setvalue)
    {
        vel = setvalue;
    }
    void changefacing(glm::mat4 setvalue)
    {
        rot = setvalue;
    }
    float gethitboxr()
    {
        return hitboxr;
    }
    bool getdeleteflag()
    {
        return deleteflag;
    }
    void setdeleteflag()
    {
        deleteflag = true;
    }
    void alignpos()
    {
        float xp = SKYBOX_EDGE_LEN_HALF;
        float yp = SKYBOX_EDGE_LEN_HALF;
        float zp = SKYBOX_EDGE_LEN_HALF;
        float xm = -SKYBOX_EDGE_LEN_HALF;
        float ym = -SKYBOX_EDGE_LEN_HALF;
        float zm = -SKYBOX_EDGE_LEN_HALF;

        if (pos.x > xp) pos.x -= (xp - xm);
        else if (pos.x < xm) pos.x += (xp - xm);
        if (pos.y > yp) pos.y -= (yp - ym);
        else if (pos.y < xm) pos.y += (yp - ym);
        if (pos.z > zp) pos.z -= (zp - zm);
        else if (pos.z < zm) pos.z += (zp - zm);
    }
protected:
    glm::vec3 pos;
    glm::vec3 vel;
    glm::mat4 rot;
    float size;
    float hp;
    float timer;
    float hitboxr;
    bool deleteflag = false;
};

bool add_asteroid(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer, int subdiv);
class Asteroid : public Object {
    int subdiv;
    glm::mat4 rotacc;
public:
    Asteroid(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer,int subdiv) : Object(pos, vel, size, hp, timer)
    {
        this->subdiv = subdiv;
        this->hitboxr = size;
        glm::vec4 rotaccaxis = glm::vec4(0.0, 0.0, 1.0, 1.0);
        glm::mat4 tempaxisrot = glm::rotate(glm::mat4(1.0f), glm::radians(float(rand() % 360)), glm::vec3(1, 0, 0));
        tempaxisrot = glm::rotate(tempaxisrot, glm::radians(float(rand() % 360)), glm::vec3(0, 1, 0));
        tempaxisrot = glm::rotate(tempaxisrot, glm::radians(float(rand() % 360)), glm::vec3(0, 0, 1));
        rotaccaxis = rotaccaxis * tempaxisrot;
        this->rotacc = glm::rotate(glm::mat4(1.0f),glm::radians(float(10+rand()%10)/(this->size * this->size * 5)), glm::vec3(rotaccaxis.x,rotaccaxis.y, rotaccaxis.z));
    }
    int getsubdiv()
    {
        return subdiv;
    }
    void updatepos()
    {
        pos = pos + vel;
        rot = rot * rotacc;
    }
    void updateelse()
    {
    }
    bool colidecheck(std::string collidetype,Object other)
    {

        std::cout << "collide with" << collidetype <<'\n';
        if (collidetype=="Projectile" && !other.getdeleteflag() && !deleteflag)
        {
            float otherhp = other.gethp();
            other.changehp(0);
            if (other.gethp() >= 0) hp -= otherhp;
            if (hp < 0)
            {
                this->setdeleteflag();
                std::cout << "this->hp==" << hp << '\n';
                hp = 0;
                if (size > ASTEROID_SIZE_BOTTOM_CAP*2)
                {
                    for (int i=0;i<3;i++)
                    {
                        glm::vec4 randdir = glm::vec4(0, 0, 1, 1);
                        glm::mat4 randmat = glm::rotate(glm::mat4(1), glm::radians(float(rand() % 360)), glm::vec3(1, 0, 0));
                        randmat = glm::rotate(randmat, glm::radians(float(rand() % 360)), glm::vec3(0, 1, 0));
                        randmat = glm::rotate(randmat, glm::radians(float(rand() % 360)), glm::vec3(0, 0, 1));
                        randdir = randdir * randmat;
                        glm::vec3 randdir_3 = glm::vec3(randdir.x * size * 0.5, randdir.y * size * 0.5, randdir.z * size * 0.5);
                        add_asteroid(pos+randdir_3, vel+ glm::vec3(randdir.x *10/size, randdir.y *10/ size, randdir.z *10/ size), size *0.5, 1+int(size), 1, (size > 24) + (size > 48));
                    }
                }
            }
        }
        else if (collidetype == "Asteroid")
        {
            float otherhp = other.gethp();
            other.changehp(otherhp - hp);
            hp -= otherhp;
            if (hp < 0 and !deleteflag)
            {
                hp = 0;
                setdeleteflag();
            }
        }
        return 1;
    }
};

class Spaceship : public Object{
public:
    Spaceship(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer) : Object(pos, vel, size, hp, timer)
    {
    }
};

class Projectile : public Object {
public:
    Projectile(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer,int subdiv) : Object(pos, vel, size, hp, timer)
    {
        this->subdiv = subdiv;
        this->hitboxr = size;
        //glm::vec3 temprot = glm::normalize(this->vel);
        // 
        //this->rot = glm::rotate(glm::mat4(1.0), glm::);
    }
    int getsubdiv()
    {
        return subdiv;
    }
    void updatepos()
    {
        pos = pos + vel;
    }
    void updateelse()
    {
        timer = timer - 1.0 / EXPECTED_FRAME_PER_SEC;
        if (timer <= 0) setdeleteflag();
    }
    bool colidecheck(std::string collidetype, Object other)
    {
        if (collidetype == "Asteroid")
        {
            //Astedoid.collide에서 처리
            setdeleteflag();
        }
        return 1;
    }
private:
    int subdiv;
};

class Particle {
public:
    Particle(glm::vec3 pos, glm::vec3 vel, glm::vec4 rot, float size, float timer, int subdiv)
    {
        this->pos = pos;
        this->vel = vel;
        this->rot = rot;
        this->size = size;
        this->subdiv = subdiv;
        this->timer = timer;
        this->deleteflag = false;
    }

    glm::vec3 getpos()
    {
        return pos;
    }
    glm::vec3 getvel()
    {
        return vel;
    }
    float getsize()
    {
        return size;
    }
    void changepos(glm::vec3 setvalue)
    {
        pos = setvalue;
    }
    void changevel(glm::vec3 setvalue)
    {
        vel = setvalue;
    }
    bool getdeleteflag()
    {
        return deleteflag;
    }
    void setdeleteflag()
    {
        deleteflag = true;
    }

private:
    glm::vec3 pos;
    glm::vec3 vel;
    glm::vec4 rot;
    float size;
    int subdiv;
    float timer;
    bool deleteflag = false;
};

// ------------ Globals ------------
GLuint gProgram = 0;
GLuint gVao = 0, gVbo = 0;
size_t gVertexCount = 0;

glm::vec3 lightPos = glm::vec3(5.0f, 5.0f, -5.0f);

int winW = 800, winH = 600;
float angle = 0.0f;
bool keyboard_up, keyboard_down, keyboard_left, keyboard_right, keyboard_space, keyboard_shift = false;
int gamestate = 1;
float maxrot = 0;
float player_vel = 0.00;
float player_acc = 0.00;
float minspeed = 0.00;
float maxspeed = 0.05;
float pitchspeed = 0.0;
float rollspeed = 0.0;
float pitchacc = 0.0;
float rollacc = 0.0;
float pitchmaxspeed = 5;
float rollmaxspeed = 5;
float pitchmaxacc = 0.15;
float rollmaxacc = 0.15;

unsigned int Globalid = 0;
Spaceship player(glm::vec3(0,0,0),glm::vec3(0,0,0),1.0,1.0,1.0);
std::vector<Projectile> list_proj;
std::vector<Asteroid> list_asteroids;
std::vector<Particle> list_particle;
float main_weapon_cooldown = 0;

bool add_asteroid(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer, int subdiv)
{
    list_asteroids.push_back(Asteroid(pos,vel,size,hp,timer,subdiv));
    return 1;
}

bool remove_deteteflag_asteroid()
{
    int listlen = list_asteroids.size();
    Asteroid* delobj;
    for (int i = listlen - 1; i >= 0; i--)
    {
        if (list_asteroids.at(i).getdeleteflag())
        {
            list_asteroids.erase(list_asteroids.begin() + i);
        }
    }
    return 1;
}

bool remove_all_asteroid()
{
    list_asteroids.clear();
    return 1;
}

bool add_projectile(glm::vec3 pos, glm::vec3 vel, float size, float hp, float timer, int subdiv)
{
    list_proj.push_back(Projectile(pos, vel, size, hp, timer, subdiv));
    return 1;
}

bool shoot_projectile(glm::vec3 pos, glm::mat4 dir, float speed, float size, float hp, float timer)
{
    glm::vec3 proj_vel = glm::normalize(getfowardvec(dir));
    glm::vec3 proj_pos = pos + proj_vel*0.5f;
    add_projectile(proj_pos, proj_vel*speed, size, hp, timer, 1);
    std::cout << "shooted";
    return 1;
}

bool remove_deteteflag_projectile()
{
    int listlen = list_proj.size();
    Projectile* delobj;
    for (int i = listlen - 1; i >= 0; i--)
    {
        if (list_proj.at(i).getdeleteflag())
        {
            list_proj.erase(list_proj.begin() + i);
        }
    }
    return 1;
}

bool remove_all_projectile()
{
    list_proj.clear();
    return 1;
}


// ------------ Setup ------------
void setupScene() {
    //for a little test
    add_asteroid(glm::vec3(0, 100, 0), glm::vec3(0, 0, 0), 54.0, 10.0, 0, 1);
    
    

    // compile shaders
    std::string vsSrc = loadFile("vertex.glsl");
    std::string fsSrc = loadFile("fragment.glsl");
    if (vsSrc.empty() || fsSrc.empty()) {
        std::cerr << "Shaders missing (vertex.glsl, fragment.glsl)\n";
        exit(1);
    }
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());
    gProgram = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);

    // create geometry

    // VAO/VBO
    glGenVertexArrays(1, &gVao);
    glBindVertexArray(gVao);
    glGenBuffers(1, &gVbo);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo);

    // layout: location 0 = position, location 1 = normal
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));

    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

// ------------ Render ------------
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(gProgram);

    GLint loc;
    glm::mat4 proj;
    glm::mat4 view;
    glm::mat4 model;
    glm::mat4 mvp;
    glm::mat3 normalMat;


    glm::mat4 player_face = player.getfacing();
    glm::vec4 foward = glm::normalize(player_face*glm::vec4(0.0, 0.0, 1.0, 1.0));
    glm::vec4 upward = glm::normalize(player_face*glm::vec4(0.0, 1.0, 0.0, 1.0));
    proj = glm::perspective(glm::radians(45.0f), (float)winW / winH, 0.1f, 400.0f);
    glm::vec3 cameraPos = player.getpos() + glm::vec3(-20 * foward.x + upward.x*3, -20 * foward.y + upward.y*3, -20 * foward.z + upward.z*3);
    glm::vec3 focusPos = player.getpos() + glm::vec3(upward.x * 3, upward.y * 3, upward.z * 3);
    view = glm::lookAt(cameraPos, focusPos, glm::vec3(upward.x, upward.y, upward.z));

    auto verts = make_player_char(player.getsize());
    gVertexCount = verts.size();
    //player loop image 만들다 말음
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

    // camera & transforms (glm)
    glm::vec3 player_altpos = altposxyz(player.getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
    glm::vec3 player_altpos1 = altposxyzp(player.getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
    glm::vec3 player_altpos2 = altposxyzm(player.getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
    glm::vec3 render_pos = player.getpos();
    
    for(int j=0;j<64;j++)
    {
        render_pos = player.getpos();
        if (j % 4 == 0) render_pos.x = player_altpos1.x;
        else if (j % 4 == 1) render_pos.x = player_altpos2.x;
        else if (j % 4 == 1) render_pos.x = player_altpos.x;
        if (int(j / 4) % 4 == 0) render_pos.y = player_altpos1.y;
        else if (int(j / 4) % 4 == 1) render_pos.y = player_altpos2.y;
        else if (int(j / 4) % 4 == 1) render_pos.y = player_altpos.y;
        if (int(j / 16) % 4 == 0) render_pos.z = player_altpos1.z;
        else if (int(j / 16) % 4 == 1) render_pos.z = player_altpos2.z;
        else if (int(j / 16) % 4 == 1) render_pos.z = player_altpos.z;

        model = glm::mat4(1.0f);
        model = glm::translate(model, render_pos);
        model = model * player.getfacing();

        mvp = proj * view * model;
        normalMat = glm::transpose(glm::inverse(glm::mat3(model)));

        loc = glGetUniformLocation(gProgram, "uMVP"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mvp));
        loc = glGetUniformLocation(gProgram, "uModel"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(model));
        loc = glGetUniformLocation(gProgram, "uNormalMat"); if (loc >= 0) glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(normalMat));
        loc = glGetUniformLocation(gProgram, "uLightPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(lightPos));
        loc = glGetUniformLocation(gProgram, "uViewPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(cameraPos));
        // material/uniform color
        loc = glGetUniformLocation(gProgram, "uColor"); if (loc >= 0) glUniform3f(loc, 1.0f, 1.0f, 1.0f);
        loc = glGetUniformLocation(gProgram, "render_type"); if (loc >= 0) glUniform1i(loc, 0);

        glBindVertexArray(gVao);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)gVertexCount);
        glBindVertexArray(0);
    }

    Asteroid* ast;
    for (int i=0;i<int(list_asteroids.size());i++)
    {
        if (true)   //typeid(list_enemies[i]).name=="Asteroid"
        {
            ast = &(list_asteroids[i]);
            glm::vec3 ast_altpos = altposxyz(ast->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            glm::vec3 ast_altpos1 = altposxyzp(ast->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            glm::vec3 ast_altpos2 = altposxyzm(ast->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            verts = make_icosphere(ast->getsubdiv());
            gVertexCount = verts.size();
            glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

            for (int j=0;j<64;j++)
            {
                glm::vec3 render_pos = ast->getpos();
                if (j % 4 == 0) render_pos.x = ast_altpos1.x;
                else if (j % 4 == 1) render_pos.x = ast_altpos2.x;
                else if (j % 4 == 2) render_pos.x = ast_altpos.x;
                if (int(j/4) % 4 == 0) render_pos.y = ast_altpos1.y;
                else if (int(j / 4) % 4 == 1) render_pos.y = ast_altpos2.y;
                else if (int(j / 4) % 4 == 2) render_pos.y = ast_altpos.y;
                if (int(j/16) % 4 == 0) render_pos.z = ast_altpos1.z;
                else if (int(j / 16) % 4 == 1) render_pos.z = ast_altpos2.z;
                else if (int(j / 16) % 4 == 2) render_pos.z = ast_altpos.z;

                // camera & transforms (glm)
                model = glm::mat4(1.0f);
                model = glm::translate(model, render_pos);
                model = model * ast->getfacing();
                model = glm::scale(model, glm::vec3(ast->getsize()));
                mvp = proj * view * model;

                normalMat = glm::transpose(glm::inverse(glm::mat3(model)));

                loc = glGetUniformLocation(gProgram, "uMVP"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mvp));
                loc = glGetUniformLocation(gProgram, "uModel"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(model));
                loc = glGetUniformLocation(gProgram, "uNormalMat"); if (loc >= 0) glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(normalMat));
                loc = glGetUniformLocation(gProgram, "uLightPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(lightPos));
                loc = glGetUniformLocation(gProgram, "uViewPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(cameraPos));
                // material/uniform color
                loc = glGetUniformLocation(gProgram, "uColor"); if (loc >= 0) glUniform3f(loc, 1.0f, 1.0f, 1.0f);
                loc = glGetUniformLocation(gProgram, "render_type"); if (loc >= 0) glUniform1i(loc, 0);

                glBindVertexArray(gVao);
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)gVertexCount);
                glBindVertexArray(0);
            }
        }
    }

    verts = make_icosphere(1);
    Projectile* prj;
    for (int i = 0; i<int(list_proj.size()); i++)
    {
        if (true)   //typeid(list_enemies[i]).name=="Projectile"
        {
            prj = &(list_proj[i]);
            glm::vec3 prj_altpos = altposxyz(prj->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            glm::vec3 prj_altpos1 = altposxyzp(prj->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            glm::vec3 prj_altpos2 = altposxyzm(prj->getpos(), player.getpos() + glm::vec3(-10 * foward.x + upward.x, -10 * foward.y + upward.y, -10 * foward.z + upward.z));
            auto verts = make_icosphere(prj->getsubdiv());
            gVertexCount = verts.size();
            glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
            for (int j = 0; j < 64; j++)
            {
                glm::vec3 render_pos = prj->getpos();
                if (j % 4 == 0) render_pos.x = prj_altpos1.x;
                else if (j % 4 == 1) render_pos.x = prj_altpos2.x;
                else if (j % 4 == 2) render_pos.x = prj_altpos.x;
                if (int(j / 4) % 4 == 0) render_pos.y = prj_altpos1.y;
                else if (int(j / 4) % 4 == 1) render_pos.y = prj_altpos2.y;
                else if (int(j / 4) % 4 == 2) render_pos.y = prj_altpos.y;
                if (int(j / 16) % 4 == 0) render_pos.z = prj_altpos1.z;
                else if (int(j / 16) % 4 == 1) render_pos.z = prj_altpos2.z;
                else if (int(j / 16) % 4 == 2) render_pos.z = prj_altpos.z;

                // camera & transforms (glm)
                model = glm::mat4(1.0f);
                model = glm::translate(model, render_pos);
                model = model * prj->getfacing();
                model = glm::scale(model, glm::vec3(prj->getsize()));
                mvp = proj * view * model;

                normalMat = glm::transpose(glm::inverse(glm::mat3(model)));

                loc = glGetUniformLocation(gProgram, "uMVP"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mvp));
                loc = glGetUniformLocation(gProgram, "uModel"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(model));
                loc = glGetUniformLocation(gProgram, "uNormalMat"); if (loc >= 0) glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(normalMat));
                loc = glGetUniformLocation(gProgram, "uLightPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(lightPos));
                loc = glGetUniformLocation(gProgram, "uViewPos"); if (loc >= 0) glUniform3fv(loc, 1, glm::value_ptr(cameraPos));
                // material/uniform color
                loc = glGetUniformLocation(gProgram, "uColor"); if (loc >= 0) glUniform3f(loc, 1.0f, 5.0f, 5.0f);
                loc = glGetUniformLocation(gProgram, "render_type"); if (loc >= 0) glUniform1i(loc, 1);

                glBindVertexArray(gVao);
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)gVertexCount);
                glBindVertexArray(0);
            }
        }
    }

    glutSwapBuffers();
}

void idle(int) {
    if (gamestate==1)
    {
        remove_deteteflag_asteroid();
        remove_deteteflag_projectile();

        Asteroid* ast;
        for (int i = 0; i<int(list_asteroids.size()); i++)
        {
            ast = &(list_asteroids[i]);
            ast->alignpos();
            ast->updatepos();
        }
        Projectile* prj;
        for (int i = 0; i<int(list_proj.size()); i++)
        {
            prj = &(list_proj[i]);
            prj->alignpos();
            prj->updatepos();
            prj->updateelse();
        }


        for (int i = 0; i<int(list_proj.size()); i++)
        {
            prj = &(list_proj[i]);
            for (int i = 0; i<int(list_asteroids.size()); i++)
            {
                ast = &(list_asteroids[i]);
                glm::vec3 temp_s = prj->getvel();
                glm::vec3 temp_1 = prj->getpos();
                glm::vec3 temp_2 = ast->getpos();
                bool hit_bool = collided_displacement(
                    temp_s.x, temp_s.y, temp_s.z,
                    temp_1.x, temp_1.y, temp_1.z,
                    prj->gethitboxr(),
                    temp_2.x, temp_2.y, temp_2.z,
                    ast->gethitboxr()
                );
                if (hit_bool)
                {
                    ast->colidecheck("Projectile", *prj);
                    prj->colidecheck("Asteroid", *ast);
                }
            }
        }

        if (keyboard_up) pitchacc += 0.005;
        else if (keyboard_down) pitchacc -= 0.005;
        else pitchacc *= 0.8;
        if (keyboard_right) rollacc += 0.005;
        else if (keyboard_left) rollacc -= 0.005;
        else rollacc *= 0.8;
        if (keyboard_shift) player_acc += 0.001;
        else player_acc = 0;
        if (keyboard_space)
        {
            if (main_weapon_cooldown <= 0)
            {
                shoot_projectile(player.getpos(), player.getfacing(), 15, 1, 1, 1.0);
                main_weapon_cooldown = MAIN_WEAPON_COOLDOWN;
            }
        }
        
        rollspeed += rollacc;
        pitchspeed += pitchacc;
        player_vel += player_acc;
        main_weapon_cooldown -= float(1.0f / EXPECTED_FRAME_PER_SEC);
        if (player_vel > maxspeed) player_vel = maxspeed;
        
        if (rollspeed > rollmaxspeed) rollspeed = rollmaxspeed;
        else if(rollspeed < -rollmaxspeed) rollspeed = -rollmaxspeed;
        if (pitchspeed > pitchmaxspeed) pitchspeed = pitchmaxspeed;
        else if (pitchspeed < -pitchmaxspeed) pitchspeed = -pitchmaxspeed;
        if (rollacc > rollmaxacc) rollacc = rollmaxacc;
        else if (rollacc < -rollmaxacc) rollacc = -rollmaxacc;
        if (pitchacc > pitchmaxacc) pitchacc = pitchmaxacc;
        else if (pitchacc < -pitchmaxacc) pitchacc = -pitchmaxacc;

        glm::mat4 player_face=player.getfacing();
        glm::mat4 xrotator = glm::rotate(glm::mat4(1.0f), glm::radians(pitchspeed), glm::vec3(1,0,0));
        player.changefacing(player_face* xrotator);
        player_face = player.getfacing();
        glm::mat4 zrotator = glm::rotate(glm::mat4(1.0f), glm::radians(rollspeed), glm::vec3(0, 0, 1));
        player.changefacing(player_face*zrotator);
        rollspeed *= 0.9;
        pitchspeed *= 0.9;

        player_face=player.getfacing();
        glm::vec3 oldvel = player.getvel();
        player_face = glm::scale(player_face, glm::vec3(-1.0, -1.0, 1.0));
        glm::vec4 mover = player_face*glm::vec4(0.0, 0.0, 1.0, 1.0);
        mover = glm::normalize(mover) * player_acc;
        glm::vec3 newvel = glm::vec3(oldvel.x*LOSE_SPEED_VAR + mover.x, oldvel.y * LOSE_SPEED_VAR + mover.y, oldvel.z * LOSE_SPEED_VAR + mover.z);
        player.changevel(newvel);
        glm::vec3 oldpos=player.getpos();
        glm::vec3 newpos = oldpos + newvel;
        player.changepos(newpos);
        player.alignpos();
    }
    glutPostRedisplay();
    glutTimerFunc(1000 / EXPECTED_FRAME_PER_SEC, idle, 1);
}

void reshape(int w, int h) {
    winW = w; winH = h;
    glViewport(0, 0, w, h);
}

void keyboard(unsigned char key, int, int)
{
    switch (key)
    {
    case ' ':
        {
            keyboard_space = true;
            break;
        }
    }
}

void keyboardup(unsigned char key, int, int)
{
    switch (key)
    {
    case ' ':
        {
        keyboard_space = false;
        break;
        }
    }
}

void special(int key, int, int)
{
    switch(key)
    {
    case 100:
        {
        keyboard_left = true;
        break;
        }
    case 101:
        {
        keyboard_up = true;
        break;
        }
    case 102:
        {
        keyboard_right = true;
        break;
        }
    case 103:
        {
        keyboard_down = true;
        break;
        }
    case 112:
        {
        keyboard_shift = true;
        break;
        }
    }
}
void specialup(int key, int, int)
{
    switch (key)
    {
    case 100:
        {
        keyboard_left = false;
        break;
        }
    case 101:
        {
        keyboard_up = false;
        break;
        }
    case 102:
        {
        keyboard_right = false;
        break;
        }
    case 103:
        {
        keyboard_down = false;
        break;
        }
    case 112:
        {
        keyboard_shift = false;
        break;
        }
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("Octahedron - flat shading (GLSL)");
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW init failed\n"; return -1;
    }
    setupScene();
    glutDisplayFunc(display);
    glutTimerFunc(1000 / EXPECTED_FRAME_PER_SEC,idle,1);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardup);
    glutSpecialFunc(special);
    glutSpecialUpFunc(specialup);
    //glutNormalKeyFunc(normalkey);
    glutSpecialUpFunc(specialup);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
