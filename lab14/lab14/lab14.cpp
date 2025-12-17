#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics/Image.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ======================= LOGS =======================
void ShaderLog(GLuint shader)
{
    GLint infologLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
        std::cout << "Shader log:\n" << infoLog.data() << std::endl;
    }
}

void ProgramLog(GLuint prog)
{
    GLint infologLen = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetProgramInfoLog(prog, infologLen, &charsWritten, infoLog.data());
        std::cout << "Program log:\n" << infoLog.data() << std::endl;
    }
}

GLuint CompileShader(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) ShaderLog(sh);
    return sh;
}

GLuint LinkProgram(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) ProgramLog(prog);
    return prog;
}

// ======================= MATH =======================
struct Vec3
{
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float X, float Y, float Z) :x(X), y(Y), z(Z) {}
};
struct Vec2
{
    float x = 0, y = 0;
    Vec2() = default;
    Vec2(float X, float Y) :x(X), y(Y) {}
};

Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x,a.y + b.y,a.z + b.z }; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x,a.y - b.y,a.z - b.z }; }
Vec3 operator*(const Vec3& a, float s) { return { a.x * s,a.y * s,a.z * s }; }

float Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float Length(const Vec3& v) { return std::sqrt(Dot(v, v)); }

Vec3 Normalize(const Vec3& v)
{
    float len = Length(v);
    if (len < 1e-6f) return v;
    return v * (1.0f / len);
}

// 4x4 column-major
struct Mat4
{
    float m[16]{};

    static Mat4 Identity()
    {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Translation(float x, float y, float z)
    {
        Mat4 r = Identity();
        r.m[12] = x; r.m[13] = y; r.m[14] = z;
        return r;
    }

    static Mat4 Scale(float x, float y, float z)
    {
        Mat4 r{};
        r.m[0] = x; r.m[5] = y; r.m[10] = z; r.m[15] = 1.0f;
        return r;
    }

    static Mat4 RotationY(float a)
    {
        Mat4 r = Identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0] = c;  r.m[2] = s;
        r.m[8] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 RotationX(float a)
    {
        Mat4 r = Identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[5] = c;  r.m[6] = -s;
        r.m[9] = s;  r.m[10] = c;
        return r;
    }

    static Mat4 Perspective(float fovyRad, float aspect, float zNear, float zFar)
    {
        Mat4 r{};
        float t = std::tan(fovyRad / 2.0f);
        r.m[0] = 1.0f / (aspect * t);
        r.m[5] = 1.0f / t;
        r.m[10] = -(zFar + zNear) / (zFar - zNear);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
        return r;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
    {
        Vec3 f = Normalize(center - eye);
        Vec3 s = Normalize(Cross(f, up));
        Vec3 u = Cross(s, f);

        Mat4 r = Identity();
        r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
        r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;

        r.m[12] = -Dot(s, eye);
        r.m[13] = -Dot(u, eye);
        r.m[14] = Dot(f, eye);
        return r;
    }
};

Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            r.m[col * 4 + row] =
            a.m[0 * 4 + row] * b.m[col * 4 + 0] +
            a.m[1 * 4 + row] * b.m[col * 4 + 1] +
            a.m[2 * 4 + row] * b.m[col * 4 + 2] +
            a.m[3 * 4 + row] * b.m[col * 4 + 3];
    return r;
}

// ======================= TEXTURE =======================
GLuint LoadTextureFromFile(const std::string& filename)
{
    sf::Image img;
    if (!img.loadFromFile(filename))
    {
        std::cout << "Failed to load texture: " << filename << "\n";
        return 0;
    }
    img.flipVertically();

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        img.getSize().x, img.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, img.getPixelsPtr());
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

// ======================= OBJ LOADER (v/vt/vn) =======================
// interleaved: pos(3) uv(2) norm(3) = 8 floats
static bool ParseFaceVertex(const std::string& s, int& vi, int& ti, int& ni)
{
    vi = ti = ni = 0;
    // formats: v, v/vt, v//vn, v/vt/vn
    size_t p1 = s.find('/');
    if (p1 == std::string::npos)
    {
        vi = std::stoi(s);
        return true;
    }
    std::string a = s.substr(0, p1);
    if (!a.empty()) vi = std::stoi(a);

    size_t p2 = s.find('/', p1 + 1);
    if (p2 == std::string::npos)
    {
        std::string b = s.substr(p1 + 1);
        if (!b.empty()) ti = std::stoi(b);
        return true;
    }

    std::string b = s.substr(p1 + 1, p2 - (p1 + 1));
    std::string c = s.substr(p2 + 1);
    if (!b.empty()) ti = std::stoi(b);
    if (!c.empty()) ni = std::stoi(c);
    return true;
}

bool LoadOBJ_PosUVNorm(const std::string& filename, std::vector<float>& out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Failed to open OBJ: " << filename << "\n";
        return false;
    }

    std::vector<Vec3> pos;
    std::vector<Vec2> uv;
    std::vector<Vec3> nrm;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            float x, y, z; iss >> x >> y >> z;
            pos.emplace_back(x, y, z);
        }
        else if (prefix == "vt")
        {
            float u, v; iss >> u >> v;
            uv.emplace_back(u, v);
        }
        else if (prefix == "vn")
        {
            float x, y, z; iss >> x >> y >> z;
            nrm.emplace_back(x, y, z);
        }
        else if (prefix == "f")
        {
            std::vector<std::string> tok;
            std::string t;
            while (iss >> t) tok.push_back(t);
            if (tok.size() < 3) continue;

            auto push = [&](int vi, int ti, int ni)
                {
                    if (vi <= 0 || vi > (int)pos.size()) return;
                    Vec3 p = pos[vi - 1];

                    Vec2 tc(0, 0);
                    if (ti > 0 && ti <= (int)uv.size()) tc = uv[ti - 1];

                    Vec3 nn(0, 1, 0);
                    if (ni > 0 && ni <= (int)nrm.size()) nn = nrm[ni - 1];

                    out.push_back(p.x); out.push_back(p.y); out.push_back(p.z);
                    out.push_back(tc.x); out.push_back(tc.y);
                    out.push_back(nn.x); out.push_back(nn.y); out.push_back(nn.z);
                };
            
            int v0, t0, n0;
            ParseFaceVertex(tok[0], v0, t0, n0);
            for (size_t i = 1; i + 1 < tok.size(); ++i)
            {
                int v1, t1, n1, v2, t2, n2;
                ParseFaceVertex(tok[i], v1, t1, n1);
                ParseFaceVertex(tok[i + 1], v2, t2, n2);

                push(v0, t0, n0);
                push(v1, t1, n1);
                push(v2, t2, n2);
            }
        }
    }

    if (out.empty())
    {
        std::cout << "OBJ has no vertices: " << filename << "\n";
        return false;
    }

    std::cout << "OBJ loaded: " << filename << ", verts: " << (out.size() / 8) << "\n";

    // Проверка: если нормалей нет (все нулевые), лучше сразу ругнуться
    bool hasNormals = false;
    for (size_t i = 0; i < out.size(); i += 8)
    {
        float nx = out[i + 5], ny = out[i + 6], nz = out[i + 7];
        if (std::fabs(nx) + std::fabs(ny) + std::fabs(nz) > 1e-6f) { hasNormals = true; break; }
    }
    if (!hasNormals)
    {
        std::cout << "WARNING: normals are missing or zero in OBJ: " << filename
            << " (need vn for correct lighting)\n";
    }

    return true;
}

// ======================= MESH =======================
struct Mesh
{
    GLuint VAO = 0, VBO = 0;
    GLsizei vertexCount = 0;
};

Mesh CreateMesh_PUVN(const std::vector<float>& data)
{
    Mesh m;
    m.vertexCount = (GLsizei)(data.size() / 8);

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

    // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // uv
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // normal
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return m;
}

// ======================= SCENE =======================
enum ShadingModel : int { SH_PHONG = 0, SH_TOON = 1, SH_GOOCH = 2 };

struct SceneObject
{
    Mesh mesh{};
    GLuint texture = 0;
    Mat4 model = Mat4::Identity();
    ShadingModel shading = SH_PHONG;

    // материал (можно настраивать отдельно)
    float ka = 0.08f;   // ambient strength
    float kd = 1.0f;    // diffuse strength
    float ks = 0.6f;    // specular strength
    float shininess = 32.0f;
};

static std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool LoadScene(const std::string& filename, std::vector<SceneObject>& outObjects)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Failed to open scene: " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);

        std::string objPath, texPath, shadingStr;
        float px, py, pz, ryDeg, sx, sy, sz;

        if (!(iss >> objPath >> texPath >> px >> py >> pz >> ryDeg >> sx >> sy >> sz >> shadingStr))
            continue;

        std::vector<float> data;
        if (!LoadOBJ_PosUVNorm(objPath, data))
        {
            std::cout << "Skip object (obj load failed): " << objPath << "\n";
            continue;
        }

        SceneObject o;
        o.mesh = CreateMesh_PUVN(data);
        o.texture = LoadTextureFromFile(texPath);
        if (!o.texture)
        {
            std::cout << "Skip object (tex load failed): " << texPath << "\n";
            // cleanup mesh
            glDeleteBuffers(1, &o.mesh.VBO);
            glDeleteVertexArrays(1, &o.mesh.VAO);
            continue;
        }

        float ry = ryDeg * (float)M_PI / 180.0f;
        o.model = Mat4::Translation(px, py, pz) * Mat4::RotationY(ry) * Mat4::Scale(sx, sy, sz);

        shadingStr = ToLower(shadingStr);
        if (shadingStr == "phong") o.shading = SH_PHONG;
        else if (shadingStr == "toon") o.shading = SH_TOON;
        else if (shadingStr == "gooch") o.shading = SH_GOOCH;
        else o.shading = SH_PHONG;

        outObjects.push_back(o);
    }

    if (outObjects.size() < 5)
    {
        std::cout << "WARNING: scene has less than 5 objects (" << outObjects.size() << ")\n";
    }

    std::cout << "Scene loaded: " << outObjects.size() << " objects\n";
    return !outObjects.empty();
}

// ======================= SHADERS =======================
const char* vertexShaderSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aNormal;

uniform mat4 uModel;//позиция/поворот/масштаб объекта
uniform mat4 uView;//положение камеры
uniform mat4 uProj;//перспектива

out vec2 vUV;
out vec3 vWorldPos;
out vec3 vWorldN;

void main()
{
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;

    // нормаль в мир: mat3(model) * normal (потом normalize)
    vWorldN = mat3(uModel) * aNormal;

    vUV = aUV;
    gl_Position = uProj * uView * wp;
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core

in vec2 vUV;
in vec3 vWorldPos;
in vec3 vWorldN;

out vec4 FragColor;

uniform sampler2D uTex;

uniform vec3 uViewPos;

// ===== lights =====
uniform int uEnablePoint;
uniform vec3 uPointPos;
uniform vec3 uPointColor;
uniform float uPointIntensity;

uniform int uEnableDir;
uniform vec3 uDirDir;      // direction TO light? обычно direction света = направление лучей (куда светит)
uniform vec3 uDirColor;
uniform float uDirIntensity;

uniform int uEnableSpot;
uniform vec3 uSpotPos;
uniform vec3 uSpotDir;
uniform vec3 uSpotColor;
uniform float uSpotIntensity;
uniform float uSpotInnerCos; // cos(innerAngle)
uniform float uSpotOuterCos; // cos(outerAngle)

// ===== material =====
uniform float uKa;
uniform float uKd;
uniform float uKs;
uniform float uShininess;

// 0=Phong, 1=Toon, 2=Gooch
uniform int uShadingModel;

vec3 ApplyPhong(vec3 baseColor, vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity)
{
    float ndotl = max(dot(N,L), 0.0);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V,R), 0.0), uShininess);

    vec3 ambient = uKa * baseColor;
    vec3 diffuse = uKd * ndotl * baseColor;
    vec3 specular = uKs * spec * vec3(1.0);

    return (ambient + diffuse + specular) * lightColor * intensity;
}

vec3 ApplyToon(vec3 baseColor, vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity)
{
    float ndotl = max(dot(N,L), 0.0);

    // ступени
    float levels = 4.0;
    float q = floor(ndotl * levels) / (levels - 1.0);

    // простой toon spec
    vec3 H = normalize(L + V);
    float s = pow(max(dot(N,H),0.0), uShininess);
    float specStep = step(0.5, s); // жёсткий блик

    vec3 ambient = 0.12 * baseColor;
    vec3 diffuse = q * baseColor;
    vec3 specular = 0.35 * specStep * vec3(1.0);

    return (ambient + diffuse + specular) * lightColor * intensity;
}

// Gooch shading (cool-to-warm)
vec3 ApplyGooch(vec3 baseColor, vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity)
{
    float ndotl = max(dot(N,L), 0.0);

    vec3 cool = vec3(0.0, 0.0, 0.55) + 0.25 * baseColor;
    vec3 warm = vec3(0.55, 0.55, 0.0) + 0.25 * baseColor;

    vec3 gooch = mix(cool, warm, ndotl);

    // мягкий spec (не Blinn: возьмём reflect как в Phong)
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V,R),0.0), uShininess);
    vec3 specular = 0.25 * spec * vec3(1.0);

    return (gooch + specular) * lightColor * intensity;
}

vec3 Shade(vec3 baseColor, vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity)
{
    if(uShadingModel==1) return ApplyToon(baseColor,N,V,L,lightColor,intensity);
    if(uShadingModel==2) return ApplyGooch(baseColor,N,V,L,lightColor,intensity);
    return ApplyPhong(baseColor,N,V,L,lightColor,intensity);
}

void main()
{
    vec3 baseColor = texture(uTex, vUV).rgb;
    vec3 N = normalize(vWorldN);//нормаль
    vec3 V = normalize(uViewPos - vWorldPos);//направление на камеру

    vec3 result = vec3(0.0);

    // ---- Point ----
    if(uEnablePoint==1)
    {
        vec3 L = normalize(uPointPos - vWorldPos);
        float dist = length(uPointPos - vWorldPos);
        float atten = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);
        result += Shade(baseColor, N, V, L, uPointColor, uPointIntensity * atten);
    }

    // ---- Directional ----
    if(uEnableDir==1)
    {
        // uDirDir пусть будет "куда светит" (направление лучей), значит L = направление к источнику = -uDirDir
        vec3 L = normalize(-uDirDir);
        result += Shade(baseColor, N, V, L, uDirColor, uDirIntensity);
    }

    // ---- Spot ----
    if(uEnableSpot==1)
    {
        vec3 L = normalize(uSpotPos - vWorldPos);
        float dist = length(uSpotPos - vWorldPos);
        float atten = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);

        // угол: сравниваем направление прожектора и направление на фрагмент
        vec3 spotDirN = normalize(uSpotDir);
        float cosTheta = dot(normalize(vWorldPos - uSpotPos), spotDirN); // направление от источника к фрагменту
        float spotFactor = clamp((cosTheta - uSpotOuterCos) / max(uSpotInnerCos - uSpotOuterCos, 1e-5), 0.0, 1.0);
        result += Shade(baseColor, N, V, L, uSpotColor, uSpotIntensity * atten * spotFactor);
    }

    FragColor = vec4(result, 1.0);
}
)";

// ======================= MAIN =======================
int main()
{
    setlocale(LC_ALL, "ru_RU.utf8");

    sf::Window window(
        sf::VideoMode({ 1200u, 900u }),
        "OpenGL Lighting Lab (Point/Dir/Spot + Phong/Toon/Gooch)",
        sf::Style::Default
    );
    window.setFramerateLimit(60);
    window.setActive(true);

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "glewInit failed: " << (const char*)glewGetErrorString(err) << "\n";
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    GLuint vs = CompileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = LinkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // uniforms
    GLint uModelLoc = glGetUniformLocation(prog, "uModel");
    GLint uViewLoc = glGetUniformLocation(prog, "uView");
    GLint uProjLoc = glGetUniformLocation(prog, "uProj");
    GLint uTexLoc = glGetUniformLocation(prog, "uTex");
    GLint uViewPosLoc = glGetUniformLocation(prog, "uViewPos");

    GLint uEnablePointLoc = glGetUniformLocation(prog, "uEnablePoint");
    GLint uPointPosLoc = glGetUniformLocation(prog, "uPointPos");
    GLint uPointColorLoc = glGetUniformLocation(prog, "uPointColor");
    GLint uPointIntLoc = glGetUniformLocation(prog, "uPointIntensity");

    GLint uEnableDirLoc = glGetUniformLocation(prog, "uEnableDir");
    GLint uDirDirLoc = glGetUniformLocation(prog, "uDirDir");
    GLint uDirColorLoc = glGetUniformLocation(prog, "uDirColor");
    GLint uDirIntLoc = glGetUniformLocation(prog, "uDirIntensity");

    GLint uEnableSpotLoc = glGetUniformLocation(prog, "uEnableSpot");
    GLint uSpotPosLoc = glGetUniformLocation(prog, "uSpotPos");
    GLint uSpotDirLoc = glGetUniformLocation(prog, "uSpotDir");
    GLint uSpotColorLoc = glGetUniformLocation(prog, "uSpotColor");
    GLint uSpotIntLoc = glGetUniformLocation(prog, "uSpotIntensity");
    GLint uSpotInnerLoc = glGetUniformLocation(prog, "uSpotInnerCos");
    GLint uSpotOuterLoc = glGetUniformLocation(prog, "uSpotOuterCos");

    GLint uKaLoc = glGetUniformLocation(prog, "uKa");
    GLint uKdLoc = glGetUniformLocation(prog, "uKd");
    GLint uKsLoc = glGetUniformLocation(prog, "uKs");
    GLint uShinLoc = glGetUniformLocation(prog, "uShininess");
    GLint uShadingModelLoc = glGetUniformLocation(prog, "uShadingModel");

    // ---- load scene ----
    std::vector<SceneObject> objects;
    if (!LoadScene("scene.txt", objects))
    {
        std::cout << "Scene load failed.\n";
        return 1;
    }

    // ---- camera ----
    Vec3 camPos(0.0f, 2.5f, 10.0f);
    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    float yaw = -90.0f;
    float pitch = -10.0f;

    auto deg2rad = [](float d) { return d * (float)M_PI / 180.0f; };

    auto calcFront = [&]()
        {
            float cy = std::cos(deg2rad(yaw));
            float sy = std::sin(deg2rad(yaw));
            float cp = std::cos(deg2rad(pitch));
            float sp = std::sin(deg2rad(pitch));
            Vec3 f;
            f.x = cy * cp;
            f.y = sp;
            f.z = sy * cp;
            return Normalize(f);
        };

    auto makeProj = [&](unsigned w, unsigned h)
        {
            float aspect = (h == 0) ? 1.0f : (float)w / (float)h;
            return Mat4::Perspective(deg2rad(60.0f), aspect, 0.1f, 200.0f);
        };

    Mat4 proj = makeProj(window.getSize().x, window.getSize().y);

    // ---- lights default ----
    bool enablePoint = true;
    bool enableDir = true;
    bool enableSpot = true;

    Vec3 pointPos(2.0f, 3.0f, 2.0f);
    Vec3 pointColor(1.0f, 0.95f, 0.85f);
    float pointInt = 2.2f;

    Vec3 dirDir(-0.3f, -1.0f, -0.2f); // куда светят лучи (вниз/вперёд)
    Vec3 dirColor(0.8f, 0.9f, 1.0f);
    float dirInt = 0.7f;

    Vec3 spotPos = camPos;
    Vec3 spotDir = calcFront();
    Vec3 spotColor(1.0f, 1.0f, 1.0f);
    float spotInt = 3.0f;
    float spotInnerDeg = 12.0f;
    float spotOuterDeg = 20.0f;

    // ---- time ----
    sf::Clock clock;
    float cameraSpeed = 7.0f;
    float rotSpeed = 60.0f;

    std::cout <<
        "\nControls:\n"
        "WASD/Space/LShift - move camera\n"
        "Arrows - rotate camera\n"
        "1/2/3 - toggle Point/Dir/Spot\n"
        "I/K - point light up/down, J/L - point left/right, U/O - point forward/back\n"
        "Z/X - change point intensity\n"
        "C/V - change spot cone (outer)\n"
        "Spotlight follows camera position+direction\n\n";

    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();

        while (auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>()) window.close();
            if (const auto* resized = event->getIf<sf::Event::Resized>())
            {
                glViewport(0, 0, resized->size.x, resized->size.y);
                proj = makeProj(resized->size.x, resized->size.y);
            }
        }

        // toggles (без “залипания”: примитивно, но норм для лабы)
        static bool k1 = false, k2 = false, k3 = false;
        bool p1 = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num1);
        bool p2 = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num2);
        bool p3 = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num3);
        if (p1 && !k1) enablePoint = !enablePoint;
        if (p2 && !k2) enableDir = !enableDir;
        if (p3 && !k3) enableSpot = !enableSpot;
        k1 = p1; k2 = p2; k3 = p3;

        Vec3 front = calcFront();
        Vec3 right = Normalize(Cross(front, worldUp));

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) camPos = camPos + front * (cameraSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) camPos = camPos - front * (cameraSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) camPos = camPos - right * (cameraSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) camPos = camPos + right * (cameraSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) camPos = camPos + worldUp * (cameraSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) camPos = camPos - worldUp * (cameraSpeed * dt);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))  yaw -= rotSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right)) yaw += rotSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))    pitch += rotSpeed * dt * 0.5f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down))  pitch -= rotSpeed * dt * 0.5f;
        pitch = std::clamp(pitch, -89.0f, 89.0f);

        front = calcFront();
        Mat4 view = Mat4::LookAt(camPos, camPos + front, worldUp);

        // point light move
        float ls = 4.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::I)) pointPos.y += ls * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::K)) pointPos.y -= ls * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::J)) pointPos.x -= ls * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::L)) pointPos.x += ls * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::U)) pointPos.z -= ls * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::O)) pointPos.z += ls * dt;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Z)) pointInt = std::max(0.0f, pointInt - 1.0f * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::X)) pointInt = pointInt + 1.0f * dt;

        // spotlight follows camera
        spotPos = camPos;
        spotDir = front;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C)) spotOuterDeg = std::max(2.0f, spotOuterDeg - 40.0f * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::V)) spotOuterDeg = std::min(60.0f, spotOuterDeg + 40.0f * dt);
        spotInnerDeg = std::min(spotOuterDeg - 1.0f, spotInnerDeg);

        // render
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);

        glUniformMatrix4fv(uViewLoc, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProjLoc, 1, GL_FALSE, proj.m);
        glUniform3f(uViewPosLoc, camPos.x, camPos.y, camPos.z);

        glUniform1i(uEnablePointLoc, enablePoint ? 1 : 0);
        glUniform3f(uPointPosLoc, pointPos.x, pointPos.y, pointPos.z);
        glUniform3f(uPointColorLoc, pointColor.x, pointColor.y, pointColor.z);
        glUniform1f(uPointIntLoc, pointInt);

        glUniform1i(uEnableDirLoc, enableDir ? 1 : 0);
        glUniform3f(uDirDirLoc, dirDir.x, dirDir.y, dirDir.z);
        glUniform3f(uDirColorLoc, dirColor.x, dirColor.y, dirColor.z);
        glUniform1f(uDirIntLoc, dirInt);

        glUniform1i(uEnableSpotLoc, enableSpot ? 1 : 0);
        glUniform3f(uSpotPosLoc, spotPos.x, spotPos.y, spotPos.z);
        glUniform3f(uSpotDirLoc, spotDir.x, spotDir.y, spotDir.z);
        glUniform3f(uSpotColorLoc, spotColor.x, spotColor.y, spotColor.z);
        glUniform1f(uSpotIntLoc, spotInt);

        float innerCos = std::cos(deg2rad(spotInnerDeg));
        float outerCos = std::cos(deg2rad(spotOuterDeg));
        glUniform1f(uSpotInnerLoc, innerCos);
        glUniform1f(uSpotOuterLoc, outerCos);

        glUniform1i(uTexLoc, 0);

        for (const auto& o : objects)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, o.texture);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, o.model.m);

            glUniform1f(uKaLoc, o.ka);
            glUniform1f(uKdLoc, o.kd);
            glUniform1f(uKsLoc, o.ks);
            glUniform1f(uShinLoc, o.shininess);
            glUniform1i(uShadingModelLoc, (int)o.shading);

            glBindVertexArray(o.mesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, o.mesh.vertexCount);
        }

        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        window.display();
    }

    // cleanup
    for (auto& o : objects)
    {
        glDeleteBuffers(1, &o.mesh.VBO);
        glDeleteVertexArrays(1, &o.mesh.VAO);
        glDeleteTextures(1, &o.texture);
    }
    glDeleteProgram(prog);
    return 0;
}
