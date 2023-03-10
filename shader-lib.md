
engine\src\scene\shader-lib\chunks\common\frag\decode.js
```glsl
vec3 decodeLinear(vec4 raw) {
    return raw.rgb;
}

float decodeGamma(float raw) {
    return pow(raw, 2.2);
}

vec3 decodeGamma(vec3 raw) {
    return pow(raw, vec3(2.2));
}

vec3 decodeGamma(vec4 raw) {
    return pow(raw.xyz, vec3(2.2));
}

vec3 decodeRGBM(vec4 raw) {
    vec3 color = (8.0 * raw.a) * raw.rgb;
    return color * color;
}

vec3 decodeRGBP(vec4 raw) {
    vec3 color = raw.rgb * (-raw.a * 7.0 + 8.0);
    return color * color;
}

vec3 decodeRGBE(vec4 raw) {
    if (raw.a == 0.0) {
        return vec3(0.0, 0.0, 0.0);
    } else {
        return raw.xyz * pow(2.0, raw.w * 255.0 - 128.0);
    }
}

vec4 passThrough(vec4 raw) {
    return raw;
}
```
engine\src\scene\shader-lib\chunks\common\frag\encode.js
```glsl
vec4 encodeLinear(vec3 source) {
    return vec4(source, 1.0);
}

vec4 encodeGamma(vec3 source) {
    return vec4(pow(source + 0.0000001, vec3(1.0 / 2.2)), 1.0);
}

vec4 encodeRGBM(vec3 source) { // modified RGBM
    vec4 result;
    result.rgb = pow(source.rgb, vec3(0.5));
    result.rgb *= 1.0 / 8.0;

    result.a = saturate( max( max( result.r, result.g ), max( result.b, 1.0 / 255.0 ) ) );
    result.a = ceil(result.a * 255.0) / 255.0;

    result.rgb /= result.a;
    return result;
}

vec4 encodeRGBP(vec3 source) {
    // convert incoming linear to gamma(ish)
    vec3 gamma = pow(source, vec3(0.5));

    // calculate the maximum component clamped to 1..8
    float maxVal = min(8.0, max(1.0, max(gamma.x, max(gamma.y, gamma.z))));

    // calculate storage factor
    float v = 1.0 - ((maxVal - 1.0) / 7.0);

    // round the value for storage in 8bit channel
    v = ceil(v * 255.0) / 255.0;

    return vec4(gamma / (-v * 7.0 + 8.0), v);    
}

vec4 encodeRGBE(vec3 source) {
    float maxVal = max(source.x, max(source.y, source.z));
    if (maxVal < 1e-32) {
        return vec4(0, 0, 0, 0);
    } else {
        float e = ceil(log2(maxVal));
        return vec4(source / pow(2.0, e), (e + 128.0) / 255.0);
    }
}
```
engine\src\scene\shader-lib\chunks\common\frag\envAtlas.js
```glsl
// the envAtlas is fixed at 512 pixels. every equirect is generated with 1 pixel boundary.
const float atlasSize = 512.0;
const float seamSize = 1.0 / atlasSize;

// map a normalized equirect UV to the given rectangle (taking 1 pixel seam into account).
vec2 mapUv(vec2 uv, vec4 rect) {
    return vec2(mix(rect.x + seamSize, rect.x + rect.z - seamSize, uv.x),
                mix(rect.y + seamSize, rect.y + rect.w - seamSize, uv.y));
}

// map a normalized equirect UV and roughness level to the correct atlas rect.
vec2 mapRoughnessUv(vec2 uv, float level) {
    float t = 1.0 / exp2(level);
    return mapUv(uv, vec4(0, 1.0 - t, t, t * 0.5));
}

// map shiny level UV
vec2 mapShinyUv(vec2 uv, float level) {
    float t = 1.0 / exp2(level);
    return mapUv(uv, vec4(1.0 - t, 1.0 - t, t, t * 0.5));
}
```
engine\src\scene\shader-lib\chunks\common\frag\envConst.js
```glsl
vec3 processEnvironment(vec3 color) {
    return color;
}
```
engine\src\scene\shader-lib\chunks\common\frag\envMultiply.js
```glsl
uniform float skyboxIntensity;

vec3 processEnvironment(vec3 color) {
    return color * skyboxIntensity;
}
```
engine\src\scene\shader-lib\chunks\common\frag\fixCubemapSeamsNone.js
```glsl
vec3 fixSeams(vec3 vec, float mipmapIndex) {
    return vec;
}

vec3 fixSeams(vec3 vec) {
    return vec;
}

vec3 fixSeamsStatic(vec3 vec, float invRecMipSize) {
    return vec;
}

vec3 calcSeam(vec3 vec) {
    return vec3(0);
}

vec3 applySeam(vec3 vec, vec3 seam, float scale) {
    return vec;
}
```
engine\src\scene\shader-lib\chunks\common\frag\fixCubemapSeamsStretch.js
```glsl
vec3 fixSeams(vec3 vec, float mipmapIndex) {
    vec3 avec = abs(vec);
    float scale = 1.0 - exp2(mipmapIndex) / 128.0;
    float M = max(max(avec.x, avec.y), avec.z);
    if (avec.x != M) vec.x *= scale;
    if (avec.y != M) vec.y *= scale;
    if (avec.z != M) vec.z *= scale;
    return vec;
}

vec3 fixSeams(vec3 vec) {
    vec3 avec = abs(vec);
    float scale = 1.0 - 1.0 / 128.0;
    float M = max(max(avec.x, avec.y), avec.z);
    if (avec.x != M) vec.x *= scale;
    if (avec.y != M) vec.y *= scale;
    if (avec.z != M) vec.z *= scale;
    return vec;
}

vec3 fixSeamsStatic(vec3 vec, float invRecMipSize) {
    vec3 avec = abs(vec);
    float scale = invRecMipSize;
    float M = max(max(avec.x, avec.y), avec.z);
    if (avec.x != M) vec.x *= scale;
    if (avec.y != M) vec.y *= scale;
    if (avec.z != M) vec.z *= scale;
    return vec;
}

vec3 calcSeam(vec3 vec) {
    vec3 avec = abs(vec);
    float M = max(avec.x, max(avec.y, avec.z));
    return vec3(avec.x != M ? 1.0 : 0.0,
                avec.y != M ? 1.0 : 0.0,
                avec.z != M ? 1.0 : 0.0);
}

vec3 applySeam(vec3 vec, vec3 seam, float scale) {
    return vec * (seam * -scale + vec3(1.0));
}
```
engine\src\scene\shader-lib\chunks\common\frag\fullscreenQuad.js
```glsl
varying vec2 vUv0;

uniform sampler2D source;

void main(void) {
    gl_FragColor = texture2D(source, vUv0);
}
```
engine\src\scene\shader-lib\chunks\common\frag\gamma1_0.js
```glsl
float gammaCorrectInput(float color) {
    return color;
}

vec3 gammaCorrectInput(vec3 color) {
    return color;
}

vec4 gammaCorrectInput(vec4 color) {
    return color;
}

vec3 gammaCorrectOutput(vec3 color) {
    return color;
}
```
engine\src\scene\shader-lib\chunks\common\frag\gamma2_2.js
```glsl
float gammaCorrectInput(float color) {
    return decodeGamma(color);
}

vec3 gammaCorrectInput(vec3 color) {
    return decodeGamma(color);
}

vec4 gammaCorrectInput(vec4 color) {
    return vec4(decodeGamma(color.xyz), color.w);
}

vec3 gammaCorrectOutput(vec3 color) {
#ifdef HDR
    return color;
#else
    return pow(color + 0.0000001, vec3(1.0 / 2.2));
#endif
}
```
engine\src\scene\shader-lib\chunks\common\frag\msdf.js
```glsl
uniform sampler2D texture_msdfMap;

#ifdef GL_OES_standard_derivatives
#define USE_FWIDTH
#endif

#ifdef GL2
#define USE_FWIDTH
#endif

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

float map (float min, float max, float v) {
    return (v - min) / (max - min);
}

uniform float font_sdfIntensity; // intensity is used to boost the value read from the SDF, 0 is no boost, 1.0 is max boost
uniform float font_pxrange;      // the number of pixels between inside and outside the font in SDF
uniform float font_textureWidth; // the width of the texture atlas

#ifdef UNIFORM_TEXT_PARAMETERS
uniform vec4 outline_color;
uniform float outline_thickness;
uniform vec4 shadow_color;
uniform vec2 shadow_offset;
#else
varying vec4 outline_color;
varying float outline_thickness;
varying vec4 shadow_color;
varying vec2 shadow_offset;
#endif

vec4 applyMsdf(vec4 color) {
    // sample the field
    vec3 tsample = texture2D(texture_msdfMap, vUv0).rgb;
    vec2 uvShdw = vUv0 - shadow_offset;
    vec3 ssample = texture2D(texture_msdfMap, uvShdw).rgb;
    // get the signed distance value
    float sigDist = median(tsample.r, tsample.g, tsample.b);
    float sigDistShdw = median(ssample.r, ssample.g, ssample.b);

    // smoothing limit - smaller value makes for sharper but more aliased text, especially on angles
    // too large value (0.5) creates a dark glow around the letters
    float smoothingMax = 0.2;

    #ifdef USE_FWIDTH
    // smoothing depends on size of texture on screen
    vec2 w = fwidth(vUv0);
    float smoothing = clamp(w.x * font_textureWidth / font_pxrange, 0.0, smoothingMax);
    #else
    float font_size = 16.0; // TODO fix this
    // smoothing gets smaller as the font size gets bigger
    // don't have fwidth we can approximate from font size, this doesn't account for scaling
    // so a big font scaled down will be wrong...
    float smoothing = clamp(font_pxrange / font_size, 0.0, smoothingMax);
    #endif

    float mapMin = 0.05;
    float mapMax = clamp(1.0 - font_sdfIntensity, mapMin, 1.0);

    // remap to a smaller range (used on smaller font sizes)
    float sigDistInner = map(mapMin, mapMax, sigDist);
    float sigDistOutline = map(mapMin, mapMax, sigDist + outline_thickness);
    sigDistShdw = map(mapMin, mapMax, sigDistShdw + outline_thickness);

    float center = 0.5;
    // calculate smoothing and use to generate opacity
    float inside = smoothstep(center-smoothing, center+smoothing, sigDistInner);
    float outline = smoothstep(center-smoothing, center+smoothing, sigDistOutline);
    float shadow = smoothstep(center-smoothing, center+smoothing, sigDistShdw);

    vec4 tcolor = (outline > inside) ? outline * vec4(outline_color.a * outline_color.rgb, outline_color.a) : vec4(0.0);
    tcolor = mix(tcolor, color, inside);

    vec4 scolor = (shadow > outline) ? shadow * vec4(shadow_color.a * shadow_color.rgb, shadow_color.a) : tcolor;
    tcolor = mix(scolor, tcolor, outline);
    
    return tcolor;
}
```
engine\src\scene\shader-lib\chunks\common\frag\outputTex2D.js
```glsl
varying vec2 vUv0;

uniform sampler2D source;

void main(void) {
    gl_FragColor = texture2D(source, vUv0);
}
```
engine\src\scene\shader-lib\chunks\common\frag\packDepth.js
```glsl
// Packing a float in GLSL with multiplication and mod
// http://blog.gradientstudios.com/2012/08/23/shadow-map-improvement
vec4 packFloat(float depth) {
    const vec4 bit_shift = vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0);
    const vec4 bit_mask  = vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);

    // combination of mod and multiplication and division works better
    vec4 res = mod(depth * bit_shift * vec4(255), vec4(256) ) / vec4(255);
    res -= res.xxyz * bit_mask;
    return res;
}
```
engine\src\scene\shader-lib\chunks\common\frag\reproject.js
```glsl
// This shader requires the following #DEFINEs:
//
// PROCESS_FUNC - must be one of reproject, prefilter
// DECODE_FUNC - must be one of decodeRGBM, decodeRGBE, decodeGamma or decodeLinear
// ENCODE_FUNC - must be one of encodeRGBM, encodeRGBE, encideGamma or encodeLinear
// SOURCE_FUNC - must be one of sampleCubemap, sampleEquirect, sampleOctahedral
// TARGET_FUNC - must be one of getDirectionCubemap, getDirectionEquirect, getDirectionOctahedral
//
// When filtering:
// NUM_SAMPLES - number of samples
// NUM_SAMPLES_SQRT - sqrt of number of samples

varying vec2 vUv0;

// source
#ifdef CUBEMAP_SOURCE
    uniform samplerCube sourceCube;
#else
    uniform sampler2D sourceTex;
#endif

#ifdef USE_SAMPLES_TEX
    // samples
    uniform sampler2D samplesTex;
    uniform vec2 samplesTexInverseSize;
#endif

// params:
// x - target cubemap face 0..6
// y - specular power (when prefiltering)
// z - source cubemap seam scale (0 to disable)
// w - target cubemap size for seam calc (0 to disable)
uniform vec4 params;

// params2:
// x - target image total pixels
// y - source cubemap size
uniform vec2 params2;

float targetFace() { return params.x; }
float specularPower() { return params.y; }
float sourceCubeSeamScale() { return params.z; }
float targetCubeSeamScale() { return params.w; }

float targetTotalPixels() { return params2.x; }
float sourceTotalPixels() { return params2.y; }

float PI = 3.141592653589793;

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

${decode}
${encode}

//-- supported projections

vec3 modifySeams(vec3 dir, float scale) {
    vec3 adir = abs(dir);
    float M = max(max(adir.x, adir.y), adir.z);
    return dir / M * vec3(
        adir.x == M ? 1.0 : scale,
        adir.y == M ? 1.0 : scale,
        adir.z == M ? 1.0 : scale
    );
}

vec2 toSpherical(vec3 dir) {
    return vec2(dir.xz == vec2(0.0) ? 0.0 : atan(dir.x, dir.z), asin(dir.y));
}

vec3 fromSpherical(vec2 uv) {
    return vec3(cos(uv.y) * sin(uv.x),
                sin(uv.y),
                cos(uv.y) * cos(uv.x));
}

vec3 getDirectionEquirect() {
    return fromSpherical((vec2(vUv0.x, 1.0 - vUv0.y) * 2.0 - 1.0) * vec2(PI, PI * 0.5));
}

// octahedral code, based on http://jcgt.org/published/0003/02/01
// "Survey of Efficient Representations for Independent Unit Vectors" by Cigolle, Donow, Evangelakos, Mara, McGuire, Meyer

float signNotZero(float k){
    return(k >= 0.0) ? 1.0 : -1.0;
}

vec2 signNotZero(vec2 v) {
    return vec2(signNotZero(v.x), signNotZero(v.y));
}

// Returns a unit vector. Argument o is an octahedral vector packed via octEncode, on the [-1, +1] square
vec3 octDecode(vec2 o) {
    vec3 v = vec3(o.x, 1.0 - abs(o.x) - abs(o.y), o.y);
    if (v.y < 0.0) {
        v.xz = (1.0 - abs(v.zx)) * signNotZero(v.xz);
    }
    return normalize(v);
}

vec3 getDirectionOctahedral() {
    return octDecode(vec2(vUv0.x, 1.0 - vUv0.y) * 2.0 - 1.0);
}

// Assumes that v is a unit vector. The result is an octahedral vector on the [-1, +1] square
vec2 octEncode(in vec3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xz * (1.0 / l1norm);
    if (v.y < 0.0) {
        result = (1.0 - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}

/////////////////////////////////////////////////////////////////////

#ifdef CUBEMAP_SOURCE
    vec4 sampleCubemap(vec3 dir) {
        return textureCube(sourceCube, modifySeams(dir, 1.0 - sourceCubeSeamScale()));
    }

    vec4 sampleCubemap(vec2 sph) {
    return sampleCubemap(fromSpherical(sph));
}

    vec4 sampleCubemap(vec3 dir, float mipLevel) {
        return textureCubeLodEXT(sourceCube, modifySeams(dir, 1.0 - exp2(mipLevel) * sourceCubeSeamScale()), mipLevel);
    }

    vec4 sampleCubemap(vec2 sph, float mipLevel) {
        return sampleCubemap(fromSpherical(sph), mipLevel);
    }
#else

    vec4 sampleEquirect(vec2 sph) {
        vec2 uv = sph / vec2(PI * 2.0, PI) + 0.5;
        return texture2D(sourceTex, vec2(uv.x, 1.0 - uv.y));
    }

    vec4 sampleEquirect(vec3 dir) {
        return sampleEquirect(toSpherical(dir));
    }

    vec4 sampleEquirect(vec2 sph, float mipLevel) {
        vec2 uv = sph / vec2(PI * 2.0, PI) + 0.5;
        return texture2DLodEXT(sourceTex, vec2(uv.x, 1.0 - uv.y), mipLevel);
    }

    vec4 sampleEquirect(vec3 dir, float mipLevel) {
        return sampleEquirect(toSpherical(dir), mipLevel);
    }

    vec4 sampleOctahedral(vec3 dir) {
        vec2 uv = octEncode(dir) * 0.5 + 0.5;
        return texture2D(sourceTex, vec2(uv.x, 1.0 - uv.y));
    }

    vec4 sampleOctahedral(vec2 sph) {
        return sampleOctahedral(fromSpherical(sph));
    }

    vec4 sampleOctahedral(vec3 dir, float mipLevel) {
        vec2 uv = octEncode(dir) * 0.5 + 0.5;
        return texture2DLodEXT(sourceTex, vec2(uv.x, 1.0 - uv.y), mipLevel);
    }

    vec4 sampleOctahedral(vec2 sph, float mipLevel) {
        return sampleOctahedral(fromSpherical(sph), mipLevel);
    }

#endif

vec3 getDirectionCubemap() {
    vec2 st = vUv0 * 2.0 - 1.0;
    float face = targetFace();

    vec3 vec;
    if (face == 0.0) {
        vec = vec3(1, -st.y, -st.x);
    } else if (face == 1.0) {
        vec = vec3(-1, -st.y, st.x);
    } else if (face == 2.0) {
        vec = vec3(st.x, 1, st.y);
    } else if (face == 3.0) {
        vec = vec3(st.x, -1, -st.y);
    } else if (face == 4.0) {
        vec = vec3(st.x, -st.y, 1);
    } else {
        vec = vec3(-st.x, -st.y, -1);
    }

    return normalize(modifySeams(vec, 1.0 / (1.0 - targetCubeSeamScale())));
}

mat3 matrixFromVector(vec3 n) { // frisvad
    float a = 1.0 / (1.0 + n.z);
    float b = -n.x * n.y * a;
    vec3 b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
    vec3 b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    return mat3(b1, b2, n);
}

mat3 matrixFromVectorSlow(vec3 n) {
    vec3 up = (1.0 - abs(n.y) <= 0.0000001) ? vec3(0.0, 0.0, n.y > 0.0 ? 1.0 : -1.0) : vec3(0.0, 1.0, 0.0);
    vec3 x = normalize(cross(up, n));
    vec3 y = cross(n, x);
    return mat3(x, y, n);
}

vec4 reproject() {
    if (NUM_SAMPLES <= 1) {
        // single sample
        return ENCODE_FUNC(DECODE_FUNC(SOURCE_FUNC(TARGET_FUNC())));
    } else {
        // multi sample
        vec3 t = TARGET_FUNC();
        vec3 tu = dFdx(t);
        vec3 tv = dFdy(t);

        vec3 result = vec3(0.0);
        for (float u = 0.0; u < NUM_SAMPLES_SQRT; ++u) {
            for (float v = 0.0; v < NUM_SAMPLES_SQRT; ++v) {
                result += DECODE_FUNC(SOURCE_FUNC(normalize(t +
                                                            tu * (u / NUM_SAMPLES_SQRT - 0.5) +
                                                            tv * (v / NUM_SAMPLES_SQRT - 0.5))));
            }
        }
        return ENCODE_FUNC(result / (NUM_SAMPLES_SQRT * NUM_SAMPLES_SQRT));
    }
}

vec4 unpackFloat = vec4(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0);

#ifdef USE_SAMPLES_TEX
    void unpackSample(int i, out vec3 L, out float mipLevel) {
        float u = (float(i * 4) + 0.5) * samplesTexInverseSize.x;
        float v = (floor(u) + 0.5) * samplesTexInverseSize.y;

        vec4 raw;
        raw.x = dot(texture2D(samplesTex, vec2(u, v)), unpackFloat); u += samplesTexInverseSize.x;
        raw.y = dot(texture2D(samplesTex, vec2(u, v)), unpackFloat); u += samplesTexInverseSize.x;
        raw.z = dot(texture2D(samplesTex, vec2(u, v)), unpackFloat); u += samplesTexInverseSize.x;
        raw.w = dot(texture2D(samplesTex, vec2(u, v)), unpackFloat);

        L.xyz = raw.xyz * 2.0 - 1.0;
        mipLevel = raw.w * 8.0;
    }

    // convolve an environment given pre-generated samples
    vec4 prefilterSamples() {
        // construct vector space given target direction
        mat3 vecSpace = matrixFromVectorSlow(TARGET_FUNC());

        vec3 L;
        float mipLevel;

        vec3 result = vec3(0.0);
        float totalWeight = 0.0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            unpackSample(i, L, mipLevel);
            result += DECODE_FUNC(SOURCE_FUNC(vecSpace * L, mipLevel)) * L.z;
            totalWeight += L.z;
        }

        return ENCODE_FUNC(result / totalWeight);
    }

    // unweighted version of prefilterSamples
    vec4 prefilterSamplesUnweighted() {
        // construct vector space given target direction
        mat3 vecSpace = matrixFromVectorSlow(TARGET_FUNC());

        vec3 L;
        float mipLevel;

        vec3 result = vec3(0.0);
        float totalWeight = 0.0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            unpackSample(i, L, mipLevel);
            result += DECODE_FUNC(SOURCE_FUNC(vecSpace * L, mipLevel));
        }

        return ENCODE_FUNC(result / float(NUM_SAMPLES));
    }
#endif

void main(void) {
    gl_FragColor = PROCESS_FUNC();
}
```
engine\src\scene\shader-lib\chunks\common\frag\screenDepth.js
```glsl
uniform highp sampler2D uSceneDepthMap;

#ifndef SCREENSIZE
#define SCREENSIZE
uniform vec4 uScreenSize;
#endif

#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif



#ifndef CAMERAPLANES
#define CAMERAPLANES
uniform vec4 camera_params; // 1 / camera_far,      camera_far,     camera_near,        is_ortho
#endif

#ifdef GL2
float linearizeDepth(float z) {
    if (camera_params.w == 0.0)
        return (camera_params.z * camera_params.y) / (camera_params.y + z * (camera_params.z - camera_params.y));
    else
        return camera_params.z + z * (camera_params.y - camera_params.z);
}
#else
#ifndef UNPACKFLOAT
#define UNPACKFLOAT
float unpackFloat(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0 / (256.0 * 256.0 * 256.0), 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0);
    return dot(rgbaDepth, bitShift);
}
#endif
#endif

// Retrieves rendered linear camera depth by UV
float getLinearScreenDepth(vec2 uv) {
    #ifdef GL2
        return linearizeDepth(texture2D(uSceneDepthMap, uv).r);
    #else
        return unpackFloat(texture2D(uSceneDepthMap, uv)) * camera_params.y;
    #endif
}

#ifndef VERTEXSHADER
// Retrieves rendered linear camera depth under the current pixel
float getLinearScreenDepth() {
    vec2 uv = gl_FragCoord.xy * uScreenSize.zw;
    return getLinearScreenDepth(uv);
}
#endif

// Generates linear camera depth for the given world position
float getLinearDepth(vec3 pos) {
    return -(matrix_view * vec4(pos, 1.0)).z;
}
```
engine\src\scene\shader-lib\chunks\common\frag\spherical.js
```glsl
// equirectangular helper functions
const float PI = 3.141592653589793;

vec2 toSpherical(vec3 dir) {
    return vec2(dir.xz == vec2(0.0) ? 0.0 : atan(dir.x, dir.z), asin(dir.y));
}

vec2 toSphericalUv(vec3 dir) {
    vec2 uv = toSpherical(dir) / vec2(PI * 2.0, PI) + 0.5;
    return vec2(uv.x, 1.0 - uv.y);
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingAces.js
```glsl
uniform float exposure;

vec3 toneMap(vec3 color) {
    float tA = 2.51;
    float tB = 0.03;
    float tC = 2.43;
    float tD = 0.59;
    float tE = 0.14;
    vec3 x = color * exposure;
    return (x*(tA*x+tB))/(x*(tC*x+tD)+tE);
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingAces2.js
```glsl
uniform float exposure;

// ACES approximation by Stephen Hill

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
const mat3 ACESInputMat = mat3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 toneMap(vec3 color) {
    color *= exposure;
    color = color * ACESInputMat;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);
    color = color * ACESOutputMat;

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);

    return color;
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingFilmic.js
```glsl
const float A =  0.15;
const float B =  0.50;
const float C =  0.10;
const float D =  0.20;
const float E =  0.02;
const float F =  0.30;
const float W =  11.2;

uniform float exposure;

vec3 uncharted2Tonemap(vec3 x) {
   return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 toneMap(vec3 color) {
    color = uncharted2Tonemap(color * exposure);
    vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W,W,W));
    color = color * whiteScale;

    return color;
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingHejl.js
```glsl
uniform float exposure;

vec3 toneMap(vec3 color) {
    color *= exposure;
    const float  A = 0.22, B = 0.3, C = .1, D = 0.2, E = .01, F = 0.3;
    const float Scl = 1.25;

    vec3 h = max( vec3(0.0), color - vec3(0.004) );
    return (h*((Scl*A)*h+Scl*vec3(C*B,C*B,C*B))+Scl*vec3(D*E,D*E,D*E)) / (h*(A*h+vec3(B,B,B))+vec3(D*F,D*F,D*F)) - Scl*vec3(E/F,E/F,E/F);
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingLinear.js
```glsl
uniform float exposure;

vec3 toneMap(vec3 color) {
    return color * exposure;
}
```
engine\src\scene\shader-lib\chunks\common\frag\tonemappingNone.js
```glsl
vec3 toneMap(vec3 color) {
    return color;
}
```
engine\src\scene\shader-lib\chunks\common\vert\fullscreenQuad.js
```glsl
attribute vec2 vertex_position;

varying vec2 vUv0;

void main(void)
{
    gl_Position = vec4(vertex_position, 0.5, 1.0);
    vUv0 = vertex_position.xy*0.5+0.5;
}
```
engine\src\scene\shader-lib\chunks\common\vert\msdf.js
```glsl
attribute vec3 vertex_outlineParameters;
attribute vec3 vertex_shadowParameters;

varying vec4 outline_color;
varying float outline_thickness;
varying vec4 shadow_color;
varying vec2 shadow_offset;

void unpackMsdfParams() {
    vec3 little = mod(vertex_outlineParameters, 256.);
    vec3 big = (vertex_outlineParameters - little) / 256.;

    outline_color.rb = little.xy / 255.;
    outline_color.ga = big.xy / 255.;

    // _outlineThicknessScale === 0.2
    outline_thickness = little.z / 255. * 0.2;

    little = mod(vertex_shadowParameters, 256.);
    big = (vertex_shadowParameters - little) / 256.;

    shadow_color.rb = little.xy / 255.;
    shadow_color.ga = big.xy / 255.;

    // vec2(little.z, big.z) / 127. - 1. remaps shadow offset from [0, 254] to [-1, 1]
    // _shadowOffsetScale === 0.005
    shadow_offset = (vec2(little.z, big.z) / 127. - 1.) * 0.005;
}
```
engine\src\scene\shader-lib\chunks\common\vert\skinBatchConst.js
```glsl
attribute float vertex_boneIndices;

uniform vec4 matrix_pose[BONE_LIMIT * 3];

mat4 getBoneMatrix(const in float i) {
    // read 4x3 matrix
    vec4 v1 = matrix_pose[int(3.0 * i)];
    vec4 v2 = matrix_pose[int(3.0 * i + 1.0)];
    vec4 v3 = matrix_pose[int(3.0 * i + 2.0)];

    // transpose to 4x4 matrix
    return mat4(
        v1.x, v2.x, v3.x, 0,
        v1.y, v2.y, v3.y, 0,
        v1.z, v2.z, v3.z, 0,
        v1.w, v2.w, v3.w, 1
    );
}
```
engine\src\scene\shader-lib\chunks\common\vert\skinBatchTex.js
```glsl
attribute float vertex_boneIndices;

uniform highp sampler2D texture_poseMap;
uniform vec4 texture_poseMapSize;

mat4 getBoneMatrix(const in float i) {
    float j = i * 3.0;
    float dx = texture_poseMapSize.z;
    float dy = texture_poseMapSize.w;

    float y = floor(j * dx);
    float x = j - (y * texture_poseMapSize.x);
    y = dy * (y + 0.5);

    // read elements of 4x3 matrix
    vec4 v1 = texture2D(texture_poseMap, vec2(dx * (x + 0.5), y));
    vec4 v2 = texture2D(texture_poseMap, vec2(dx * (x + 1.5), y));
    vec4 v3 = texture2D(texture_poseMap, vec2(dx * (x + 2.5), y));

    // transpose to 4x4 matrix
    return mat4(
        v1.x, v2.x, v3.x, 0,
        v1.y, v2.y, v3.y, 0,
        v1.z, v2.z, v3.z, 0,
        v1.w, v2.w, v3.w, 1
    );
}
```
engine\src\scene\shader-lib\chunks\common\vert\skinConst.js
```glsl
attribute vec4 vertex_boneWeights;
attribute vec4 vertex_boneIndices;

uniform vec4 matrix_pose[BONE_LIMIT * 3];

void getBoneMatrix(const in float i, out vec4 v1, out vec4 v2, out vec4 v3) {
    // read 4x3 matrix
    v1 = matrix_pose[int(3.0 * i)];
    v2 = matrix_pose[int(3.0 * i + 1.0)];
    v3 = matrix_pose[int(3.0 * i + 2.0)];
}

mat4 getSkinMatrix(const in vec4 indices, const in vec4 weights) {
    // get 4 bone matrices
    vec4 a1, a2, a3;
    getBoneMatrix(indices.x, a1, a2, a3);

    vec4 b1, b2, b3;
    getBoneMatrix(indices.y, b1, b2, b3);

    vec4 c1, c2, c3;
    getBoneMatrix(indices.z, c1, c2, c3);

    vec4 d1, d2, d3;
    getBoneMatrix(indices.w, d1, d2, d3);

    // multiply them by weights and add up to get final 4x3 matrix
    vec4 v1 = a1 * weights.x + b1 * weights.y + c1 * weights.z + d1 * weights.w;
    vec4 v2 = a2 * weights.x + b2 * weights.y + c2 * weights.z + d2 * weights.w;
    vec4 v3 = a3 * weights.x + b3 * weights.y + c3 * weights.z + d3 * weights.w;

    // add up weights
    float one = dot(weights, vec4(1.0));

    // transpose to 4x4 matrix
    return mat4(
        v1.x, v2.x, v3.x, 0,
        v1.y, v2.y, v3.y, 0,
        v1.z, v2.z, v3.z, 0,
        v1.w, v2.w, v3.w, one
    );
}
```
engine\src\scene\shader-lib\chunks\common\vert\skinTex.js
```glsl

attribute vec4 vertex_boneWeights;
attribute vec4 vertex_boneIndices;

uniform highp sampler2D texture_poseMap;
uniform vec4 texture_poseMapSize;

void getBoneMatrix(const in float index, out vec4 v1, out vec4 v2, out vec4 v3) {

    float i = float(index);
    float j = i * 3.0;
    float dx = texture_poseMapSize.z;
    float dy = texture_poseMapSize.w;
    
    float y = floor(j * dx);
    float x = j - (y * texture_poseMapSize.x);
    y = dy * (y + 0.5);

    // read elements of 4x3 matrix
    v1 = texture2D(texture_poseMap, vec2(dx * (x + 0.5), y));
    v2 = texture2D(texture_poseMap, vec2(dx * (x + 1.5), y));
    v3 = texture2D(texture_poseMap, vec2(dx * (x + 2.5), y));
}

mat4 getSkinMatrix(const in vec4 indices, const in vec4 weights) {
    // get 4 bone matrices
    vec4 a1, a2, a3;
    getBoneMatrix(indices.x, a1, a2, a3);

    vec4 b1, b2, b3;
    getBoneMatrix(indices.y, b1, b2, b3);

    vec4 c1, c2, c3;
    getBoneMatrix(indices.z, c1, c2, c3);

    vec4 d1, d2, d3;
    getBoneMatrix(indices.w, d1, d2, d3);

    // multiply them by weights and add up to get final 4x3 matrix
    vec4 v1 = a1 * weights.x + b1 * weights.y + c1 * weights.z + d1 * weights.w;
    vec4 v2 = a2 * weights.x + b2 * weights.y + c2 * weights.z + d2 * weights.w;
    vec4 v3 = a3 * weights.x + b3 * weights.y + c3 * weights.z + d3 * weights.w;

    // add up weights
    float one = dot(weights, vec4(1.0));

    // transpose to 4x4 matrix
    return mat4(
        v1.x, v2.x, v3.x, 0,
        v1.y, v2.y, v3.y, 0,
        v1.z, v2.z, v3.z, 0,
        v1.w, v2.w, v3.w, one
    );
}
```
engine\src\scene\shader-lib\chunks\common\vert\transform.js
```glsl
#ifdef PIXELSNAP
uniform vec4 uScreenSize;
#endif

#ifdef SCREENSPACE
uniform float projectionFlipY;
#endif

#ifdef MORPHING
uniform vec4 morph_weights_a;
uniform vec4 morph_weights_b;
#endif

#ifdef MORPHING_TEXTURE_BASED
uniform vec4 morph_tex_params;

vec2 getTextureMorphCoords() {
    float vertexId = morph_vertex_id;
    vec2 textureSize = morph_tex_params.xy;
    vec2 invTextureSize = morph_tex_params.zw;

    // turn vertexId into int grid coordinates
    float morphGridV = floor(vertexId * invTextureSize.x);
    float morphGridU = vertexId - (morphGridV * textureSize.x);

    // convert grid coordinates to uv coordinates with half pixel offset
    vec2 uv = (vec2(morphGridU, morphGridV) * invTextureSize) + (0.5 * invTextureSize);
    return getImageEffectUV(uv);
}
#endif

#ifdef MORPHING_TEXTURE_BASED_POSITION
uniform highp sampler2D morphPositionTex;
#endif

mat4 getModelMatrix() {
    #ifdef DYNAMICBATCH
    return getBoneMatrix(vertex_boneIndices);
    #elif defined(SKIN)
    return matrix_model * getSkinMatrix(vertex_boneIndices, vertex_boneWeights);
    #elif defined(INSTANCING)
    return mat4(instance_line1, instance_line2, instance_line3, instance_line4);
    #else
    return matrix_model;
    #endif
}

vec4 getPosition() {
    dModelMatrix = getModelMatrix();
    vec3 localPos = vertex_position;

    #ifdef NINESLICED
    // outer and inner vertices are at the same position, scale both
    localPos.xz *= outerScale;

    // offset inner vertices inside
    // (original vertices must be in [-1;1] range)
    vec2 positiveUnitOffset = clamp(vertex_position.xz, vec2(0.0), vec2(1.0));
    vec2 negativeUnitOffset = clamp(-vertex_position.xz, vec2(0.0), vec2(1.0));
    localPos.xz += (-positiveUnitOffset * innerOffset.xy + negativeUnitOffset * innerOffset.zw) * vertex_texCoord0.xy;

    vTiledUv = (localPos.xz - outerScale + innerOffset.xy) * -0.5 + 1.0; // uv = local pos - inner corner

    localPos.xz *= -0.5; // move from -1;1 to -0.5;0.5
    localPos = localPos.xzy;
    #endif

    #ifdef MORPHING
    #ifdef MORPHING_POS03
    localPos.xyz += morph_weights_a[0] * morph_pos0;
    localPos.xyz += morph_weights_a[1] * morph_pos1;
    localPos.xyz += morph_weights_a[2] * morph_pos2;
    localPos.xyz += morph_weights_a[3] * morph_pos3;
    #endif // MORPHING_POS03
    #ifdef MORPHING_POS47
    localPos.xyz += morph_weights_b[0] * morph_pos4;
    localPos.xyz += morph_weights_b[1] * morph_pos5;
    localPos.xyz += morph_weights_b[2] * morph_pos6;
    localPos.xyz += morph_weights_b[3] * morph_pos7;
    #endif // MORPHING_POS47
    #endif // MORPHING

    #ifdef MORPHING_TEXTURE_BASED_POSITION
    // apply morph offset from texture
    vec2 morphUV = getTextureMorphCoords();
    vec3 morphPos = texture2D(morphPositionTex, morphUV).xyz;
    localPos += morphPos;
    #endif

    vec4 posW = dModelMatrix * vec4(localPos, 1.0);
    #ifdef SCREENSPACE
    posW.zw = vec2(0.0, 1.0);
    #endif
    dPositionW = posW.xyz;

    vec4 screenPos;
    #ifdef UV1LAYOUT
    screenPos = vec4(vertex_texCoord1.xy * 2.0 - 1.0, 0.5, 1);
    #else
    #ifdef SCREENSPACE
    screenPos = posW;
    screenPos.y *= projectionFlipY;
    #else
    screenPos = matrix_viewProjection * posW;
    #endif

    #ifdef PIXELSNAP
    // snap vertex to a pixel boundary
    screenPos.xy = (screenPos.xy * 0.5) + 0.5;
    screenPos.xy *= uScreenSize.xy;
    screenPos.xy = floor(screenPos.xy);
    screenPos.xy *= uScreenSize.zw;
    screenPos.xy = (screenPos.xy * 2.0) - 1.0;
    #endif
    #endif

    return screenPos;
}

vec3 getWorldPosition() {
    return dPositionW;
}
```
engine\src\scene\shader-lib\chunks\common\vert\transformDecl.js
```glsl
attribute vec3 vertex_position;

uniform mat4 matrix_model;
uniform mat4 matrix_viewProjection;

vec3 dPositionW;
mat4 dModelMatrix;
```
engine\src\scene\shader-lib\chunks\lightmapper\frag\bakeDirLmEnd.js
```glsl
    vec4 dirLm = texture2D(texture_dirLightMap, vUv1);

    if (bakeDir > 0.5) {
        if (dAtten > 0.00001) {
            dirLm.xyz = dirLm.xyz * 2.0 - vec3(1.0);
            dAtten = saturate(dAtten);
            gl_FragColor.rgb = normalize(dLightDirNormW.xyz*dAtten + dirLm.xyz*dirLm.w) * 0.5 + vec3(0.5);
            gl_FragColor.a = dirLm.w + dAtten;
            gl_FragColor.a = max(gl_FragColor.a, 1.0 / 255.0);
        } else {
            gl_FragColor = dirLm;
        }
    } else {
        gl_FragColor.rgb = dirLm.xyz;
        gl_FragColor.a = max(dirLm.w, dAtten > 0.00001? (1.0/255.0) : 0.0);
    }
```
engine\src\scene\shader-lib\chunks\lightmapper\frag\bakeLmEnd.js
```glsl
#ifdef LIGHTMAP_RGBM
    gl_FragColor.rgb = dDiffuseLight;
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(0.5));
    gl_FragColor.rgb /= 8.0;
    gl_FragColor.a = clamp( max( max( gl_FragColor.r, gl_FragColor.g ), max( gl_FragColor.b, 1.0 / 255.0 ) ), 0.0,1.0 );
    gl_FragColor.a = ceil(gl_FragColor.a * 255.0) / 255.0;
    gl_FragColor.rgb /= gl_FragColor.a;
#else
    gl_FragColor = vec4(dDiffuseLight, 1.0);
#endif
```
engine\src\scene\shader-lib\chunks\lightmapper\frag\bilateralDeNoise.js
```glsl
// bilateral filter, based on https://www.shadertoy.com/view/4dfGDH# and
// http://people.csail.mit.edu/sparis/bf_course/course_notes.pdf

// A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images.
// It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.
// This weight can be based on a Gaussian distribution. Crucially, the weights depend not only on
// Euclidean distance of pixels, but also on the radiometric differences (e.g., range differences, such
// as color intensity, depth distance, etc.). This preserves sharp edges.

float normpdf3(in vec3 v, in float sigma) {
    return 0.39894 * exp(-0.5 * dot(v, v) / (sigma * sigma)) / sigma;
}

vec3 decodeRGBM(vec4 rgbm) {
    vec3 color = (8.0 * rgbm.a) * rgbm.rgb;
    return color * color;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec4 encodeRGBM(vec3 color) { // modified RGBM
    vec4 encoded;
    encoded.rgb = pow(color.rgb, vec3(0.5));
    encoded.rgb *= 1.0 / 8.0;

    encoded.a = saturate( max( max( encoded.r, encoded.g ), max( encoded.b, 1.0 / 255.0 ) ) );
    encoded.a = ceil(encoded.a * 255.0) / 255.0;

    encoded.rgb /= encoded.a;
    return encoded;
}

// filter size
#define MSIZE 15

varying vec2 vUv0;
uniform sampler2D source;
uniform vec2 pixelOffset;
uniform vec2 sigmas;
uniform float bZnorm;
uniform float kernel[MSIZE];

void main(void) {
    
    vec4 pixelRgbm = texture2D(source, vUv0);

    // lightmap specific optimization - skip pixels that were not baked
    // this also allows dilate filter that work on the output of this to work correctly, as it depends on .a being zero
    // to dilate, which the following blur filter would otherwise modify
    if (pixelRgbm.a <= 0.0) {
        gl_FragColor = pixelRgbm;
        return ;
    }

    // range sigma - controls blurriness based on a pixel distance
    float sigma = sigmas.x;

    // domain sigma - controls blurriness based on a pixel similarity (to preserve edges)
    float bSigma = sigmas.y;

    vec3 pixelHdr = decodeRGBM(pixelRgbm);
    vec3 accumulatedHdr = vec3(0.0);
    float accumulatedFactor = 0.0;

    // read out the texels
    const int kSize = (MSIZE-1)/2;
    for (int i = -kSize; i <= kSize; ++i) {
        for (int j = -kSize; j <= kSize; ++j) {
            
            // sample the pixel with offset
            vec2 coord = vUv0 + vec2(float(i), float(j)) * pixelOffset;
            vec4 rgbm = texture2D(source, coord);

            // lightmap - only use baked pixels
            if (rgbm.a > 0.0) {
                vec3 hdr = decodeRGBM(rgbm);

                // bilateral factors
                float factor = kernel[kSize + j] * kernel[kSize + i];
                factor *= normpdf3(hdr - pixelHdr, bSigma) * bZnorm;

                // accumulate
                accumulatedHdr += factor * hdr;
                accumulatedFactor += factor;
            }
        }
    }

    gl_FragColor = encodeRGBM(accumulatedHdr / accumulatedFactor);
}
```
engine\src\scene\shader-lib\chunks\lightmapper\frag\dilate.js
```glsl

varying vec2 vUv0;

uniform sampler2D source;
uniform vec2 pixelOffset;

void main(void) {
    vec4 c = texture2D(source, vUv0);
    c = c.a>0.0? c : texture2D(source, vUv0 - pixelOffset);
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(0, -pixelOffset.y));
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(pixelOffset.x, -pixelOffset.y));
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(-pixelOffset.x, 0));
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(pixelOffset.x, 0));
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(-pixelOffset.x, pixelOffset.y));
    c = c.a>0.0? c : texture2D(source, vUv0 + vec2(0, pixelOffset.y));
    c = c.a>0.0? c : texture2D(source, vUv0 + pixelOffset);
    gl_FragColor = c;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\ambientConstant.js
```glsl
void addAmbient() {
    dDiffuseLight += light_globalAmbient;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\ambientEnv.js
```glsl
#ifndef ENV_ATLAS
#define ENV_ATLAS
uniform sampler2D texture_envAtlas;
#endif

void addAmbient() {
    vec3 dir = normalize(cubeMapRotate(dNormalW) * vec3(-1.0, 1.0, 1.0));
    vec2 uv = mapUv(toSphericalUv(dir), vec4(128.0, 256.0 + 128.0, 64.0, 32.0) / atlasSize);

    vec4 raw = texture2D(texture_envAtlas, uv);
    vec3 linear = $DECODE(raw);
    dDiffuseLight += processEnvironment(linear);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\ambientSH.js
```glsl
uniform vec3 ambientSH[9];

void addAmbient() {
    vec3 n = cubeMapRotate(dNormalW);

    vec3 color =
        ambientSH[0] +
        ambientSH[1] * n.x +
        ambientSH[2] * n.y +
        ambientSH[3] * n.z +
        ambientSH[4] * n.x * n.z +
        ambientSH[5] * n.z * n.y +
        ambientSH[6] * n.y * n.x +
        ambientSH[7] * (3.0 * n.z * n.z - 1.0) +
        ambientSH[8] * (n.x * n.x - n.y * n.y);

    dDiffuseLight += processEnvironment(max(color, vec3(0.0)));
}
```
engine\src\scene\shader-lib\chunks\lit\frag\aoDiffuseOcc.js
```glsl
void occludeDiffuse() {
    dDiffuseLight *= dAo;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\aoSpecOcc.js
```glsl
uniform float material_occludeSpecularIntensity;

void occludeSpecular() {
    // approximated specular occlusion from AO
    float specPow = exp2(dGlossiness * 11.0);
    // http://research.tri-ace.com/Data/cedec2011_RealtimePBR_Implementation_e.pptx
    float specOcc = saturate(pow(dot(dNormalW, dViewDirW) + dAo, 0.01*specPow) - 1.0 + dAo);
    specOcc = mix(1.0, specOcc, material_occludeSpecularIntensity);

    dSpecularLight *= specOcc;
    dReflection *= specOcc;
    
#ifdef LIT_SHEEN
    sSpecularLight *= specOcc;
    sReflection *= specOcc;
#endif
}
```
engine\src\scene\shader-lib\chunks\lit\frag\aoSpecOccConst.js
```glsl
void occludeSpecular() {
    // approximated specular occlusion from AO
    float specPow = exp2(dGlossiness * 11.0);
    // http://research.tri-ace.com/Data/cedec2011_RealtimePBR_Implementation_e.pptx
    float specOcc = saturate(pow(dot(dNormalW, dViewDirW) + dAo, 0.01*specPow) - 1.0 + dAo);

    dSpecularLight *= specOcc;
    dReflection *= specOcc;
    
#ifdef LIT_SHEEN
    sSpecularLight *= specOcc;
    sReflection *= specOcc;
#endif
}
```
engine\src\scene\shader-lib\chunks\lit\frag\aoSpecOccConstSimple.js
```glsl
void occludeSpecular() {
    dSpecularLight *= dAo;
    dReflection *= dAo;

#ifdef LIT_SHEEN
    sSpecularLight *= dAo;
    sReflection *= dAo;
#endif

}
```
engine\src\scene\shader-lib\chunks\lit\frag\aoSpecOccSimple.js
```glsl
uniform float material_occludeSpecularIntensity;

void occludeSpecular() {
    float specOcc = mix(1.0, dAo, material_occludeSpecularIntensity);
    dSpecularLight *= specOcc;
    dReflection *= specOcc;

#ifdef LIT_SHEEN
    sSpecularLight *= specOcc;
    sReflection *= specOcc;
#endif
}
```
engine\src\scene\shader-lib\chunks\lit\frag\base.js
```glsl
uniform vec3 view_position;

uniform vec3 light_globalAmbient;

float square(float x) {
    return x*x;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 saturate(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}
```
engine\src\scene\shader-lib\chunks\lit\frag\baseNineSliced.js
```glsl
#define NINESLICED

varying vec2 vMask;
varying vec2 vTiledUv;

uniform mediump vec4 innerOffset;
uniform mediump vec2 outerScale;
uniform mediump vec4 atlasRect;

vec2 nineSlicedUv;
```
engine\src\scene\shader-lib\chunks\lit\frag\baseNineSlicedTiled.js
```glsl
#define NINESLICED
#define NINESLICETILED

varying vec2 vMask;
varying vec2 vTiledUv;

uniform mediump vec4 innerOffset;
uniform mediump vec2 outerScale;
uniform mediump vec4 atlasRect;

vec2 nineSlicedUv;
```
engine\src\scene\shader-lib\chunks\lit\frag\biasConst.js
```glsl
#define SHADOWBIAS

float getShadowBias(float resolution, float maxBias) {
    return maxBias;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\blurVSM.js
```glsl
varying vec2 vUv0;

uniform sampler2D source;
uniform vec2 pixelOffset;

#ifdef GAUSS
uniform float weight[SAMPLES];
#endif

#ifdef PACKED
float decodeFloatRG(vec2 rg) {
    return rg.y*(1.0/255.0) + rg.x;
}

vec2 encodeFloatRG( float v ) {
    vec2 enc = vec2(1.0, 255.0) * v;
    enc = fract(enc);
    enc -= enc.yy * vec2(1.0/255.0, 1.0/255.0);
    return enc;
}
#endif

void main(void) {
    vec3 moments = vec3(0.0);
    vec2 uv = vUv0 - pixelOffset * (float(SAMPLES) * 0.5);
    for (int i=0; i<SAMPLES; i++) {
        vec4 c = texture2D(source, uv + pixelOffset * float(i));

        #ifdef PACKED
        c.xy = vec2(decodeFloatRG(c.xy), decodeFloatRG(c.zw));
        #endif

        #ifdef GAUSS
        moments += c.xyz * weight[i];
        #else
        moments += c.xyz;
        #endif
    }

    #ifndef GAUSS
    moments /= float(SAMPLES);
    #endif

    #ifdef PACKED
    gl_FragColor = vec4(encodeFloatRG(moments.x), encodeFloatRG(moments.y));
    #else
    gl_FragColor = vec4(moments.x, moments.y, moments.z, 1.0);
    #endif
}
```
engine\src\scene\shader-lib\chunks\lit\frag\clusteredLight.js
```glsl
uniform highp sampler2D clusterWorldTexture;
uniform highp sampler2D lightsTexture8;
uniform highp sampler2D lightsTextureFloat;

// complex ifdef expression are not supported, handle it here
// defined(CLUSTER_COOKIES) || defined(CLUSTER_SHADOWS)
#if defined(CLUSTER_COOKIES)
    #define CLUSTER_COOKIES_OR_SHADOWS
#endif
#if defined(CLUSTER_SHADOWS)
    #define CLUSTER_COOKIES_OR_SHADOWS
#endif

#ifdef CLUSTER_SHADOWS
    #ifdef GL2
        // TODO: when VSM shadow is supported, it needs to use sampler2D in webgl2
        uniform sampler2DShadow shadowAtlasTexture;
    #else
        uniform sampler2D shadowAtlasTexture;
    #endif
#endif

#ifdef CLUSTER_COOKIES
    uniform sampler2D cookieAtlasTexture;
#endif

#ifdef GL2
    uniform int clusterMaxCells;
#else
    uniform float clusterMaxCells;
    uniform vec4 lightsTextureInvSize;
#endif

uniform vec3 clusterCellsCountByBoundsSize;
uniform vec3 clusterTextureSize;
uniform vec3 clusterBoundsMin;
uniform vec3 clusterBoundsDelta;
uniform vec3 clusterCellsDot;
uniform vec3 clusterCellsMax;
uniform vec2 clusterCompressionLimit0;
uniform vec2 shadowAtlasParams;

// structure storing light properties of a clustered light
// it's sorted to have all vectors aligned to 4 floats to limit padding
struct ClusterLightData {

    // area light sizes / orientation
    vec3 halfWidth;

    // type of the light (spot or omni)
    float lightType;

    // area light sizes / orientation
    vec3 halfHeight;

    #ifdef GL2
        // light index
        int lightIndex;
    #else
        // v coordinate to look up the light textures - this is the same as lightIndex but in 0..1 range
        float lightV;
    #endif

    // world space position
    vec3 position;

    // area light shape
    float shape;

    // world space direction (spot light only)
    vec3 direction;

    // light follow mode
    float falloffMode;

    // color
    vec3 color;

    // 0.0 if the light doesn't cast shadows
    float shadowIntensity;

    // atlas viewport for omni light shadow and cookie (.xy is offset to the viewport slot, .z is size of the face in the atlas)
    vec3 omniAtlasViewport;

    // range of the light
    float range;

    // channel mask - one of the channels has 1, the others are 0
    vec4 cookieChannelMask;

    // shadow bias values
    float shadowBias;
    float shadowNormalBias;

    // spot light inner and outer angle cosine
    float innerConeAngleCos;
    float outerConeAngleCos;

    // 1.0 if the light has a cookie texture
    float cookie;

    // 1.0 if cookie texture is rgb, otherwise it is using a single channel selectable by cookieChannelMask
    float cookieRgb;

    // intensity of the cookie
    float cookieIntensity;

    // light mask
    float mask;
};

// Note: on some devices (tested on Pixel 3A XL), this matrix when stored inside the light struct has lower precision compared to
// when stored outside, so we store it outside to avoid spot shadow flickering. This might need to be done to other / all members
// of the structure if further similar issues are observed.

// shadow (spot light only) / cookie projection matrix
mat4 lightProjectionMatrix;

// macros for light properties
#define isClusteredLightCastShadow(light) ( light.shadowIntensity > 0.0 )
#define isClusteredLightCookie(light) (light.cookie > 0.5 )
#define isClusteredLightCookieRgb(light) (light.cookieRgb > 0.5 )
#define isClusteredLightSpot(light) ( light.lightType > 0.5 )
#define isClusteredLightFalloffLinear(light) ( light.falloffMode < 0.5 )

// macros to test light shape
// Note: Following functions need to be called serially in listed order as they do not test both '>' and '<'
#define isClusteredLightArea(light) ( light.shape > 0.1 )
#define isClusteredLightRect(light) ( light.shape < 0.3 )
#define isClusteredLightDisk(light) ( light.shape < 0.6 )

// macro to test light mask (mesh accepts dynamic vs lightmapped lights)
#ifdef CLUSTER_MESH_DYNAMIC_LIGHTS
    // accept lights marked as dynamic or both dynamic and lightmapped
    #define acceptLightMask(light) ( light.mask < 0.75)
#else
    // accept lights marked as lightmapped or both dynamic and lightmapped
    #define acceptLightMask(light) ( light.mask > 0.25)
#endif

vec4 decodeClusterLowRange4Vec4(vec4 d0, vec4 d1, vec4 d2, vec4 d3) {
    return vec4(
        bytes2floatRange4(d0, -2.0, 2.0),
        bytes2floatRange4(d1, -2.0, 2.0),
        bytes2floatRange4(d2, -2.0, 2.0),
        bytes2floatRange4(d3, -2.0, 2.0)
    );
}

#ifdef GL2

    vec4 sampleLightsTexture8(const ClusterLightData clusterLightData, int index) {
        return texelFetch(lightsTexture8, ivec2(index, clusterLightData.lightIndex), 0);
    }

    vec4 sampleLightTextureF(const ClusterLightData clusterLightData, int index) {
        return texelFetch(lightsTextureFloat, ivec2(index, clusterLightData.lightIndex), 0);
    }

#else

    vec4 sampleLightsTexture8(const ClusterLightData clusterLightData, float index) {
        return texture2DLodEXT(lightsTexture8, vec2(index * lightsTextureInvSize.z, clusterLightData.lightV), 0.0);
    }

    vec4 sampleLightTextureF(const ClusterLightData clusterLightData, float index) {
        return texture2DLodEXT(lightsTextureFloat, vec2(index * lightsTextureInvSize.x, clusterLightData.lightV), 0.0);
    }

#endif

void decodeClusterLightCore(inout ClusterLightData clusterLightData, float lightIndex) {

    // light index
    #ifdef GL2
        clusterLightData.lightIndex = int(lightIndex);
    #else
        clusterLightData.lightV = (lightIndex + 0.5) * lightsTextureInvSize.w;
    #endif

    // shared data from 8bit texture
    vec4 lightInfo = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_FLAGS);
    clusterLightData.lightType = lightInfo.x;
    clusterLightData.shape = lightInfo.y;
    clusterLightData.falloffMode = lightInfo.z;
    clusterLightData.shadowIntensity = lightInfo.w;

    // color
    vec4 colorA = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_COLOR_A);
    vec4 colorB = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_COLOR_B);
    clusterLightData.color = vec3(bytes2float2(colorA.xy), bytes2float2(colorA.zw), bytes2float2(colorB.xy)) * clusterCompressionLimit0.y;

    // cookie
    clusterLightData.cookie = colorB.z;

    // light mask
    clusterLightData.mask = colorB.w;

    #ifdef CLUSTER_TEXTURE_FLOAT

        vec4 lightPosRange = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_POSITION_RANGE);
        clusterLightData.position = lightPosRange.xyz;
        clusterLightData.range = lightPosRange.w;

        // spot light direction
        vec4 lightDir_Unused = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_SPOT_DIRECTION);
        clusterLightData.direction = lightDir_Unused.xyz;

    #else   // 8bit

        vec4 encPosX = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_POSITION_X);
        vec4 encPosY = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_POSITION_Y);
        vec4 encPosZ = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_POSITION_Z);
        clusterLightData.position = vec3(bytes2float4(encPosX), bytes2float4(encPosY), bytes2float4(encPosZ)) * clusterBoundsDelta + clusterBoundsMin;

        vec4 encRange = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_RANGE);
        clusterLightData.range = bytes2float4(encRange) * clusterCompressionLimit0.x;

        // spot light direction
        vec4 encDirX = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_SPOT_DIRECTION_X);
        vec4 encDirY = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_SPOT_DIRECTION_Y);
        vec4 encDirZ = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_SPOT_DIRECTION_Z);
        clusterLightData.direction = vec3(bytes2float4(encDirX), bytes2float4(encDirY), bytes2float4(encDirZ)) * 2.0 - 1.0;

    #endif
}

void decodeClusterLightSpot(inout ClusterLightData clusterLightData) {

    // spot light cos angles
    vec4 coneAngle = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_SPOT_ANGLES);
    clusterLightData.innerConeAngleCos = bytes2float2(coneAngle.xy) * 2.0 - 1.0;
    clusterLightData.outerConeAngleCos = bytes2float2(coneAngle.zw) * 2.0 - 1.0;
}

void decodeClusterLightOmniAtlasViewport(inout ClusterLightData clusterLightData) {
    #ifdef CLUSTER_TEXTURE_FLOAT
        clusterLightData.omniAtlasViewport = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_PROJ_MAT_0).xyz;
    #else
        vec4 viewportA = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_ATLAS_VIEWPORT_A);
        vec4 viewportB = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_ATLAS_VIEWPORT_B);
        clusterLightData.omniAtlasViewport = vec3(bytes2float2(viewportA.xy), bytes2float2(viewportA.zw), bytes2float2(viewportB.xy));
    #endif
}

void decodeClusterLightAreaData(inout ClusterLightData clusterLightData) {
    #ifdef CLUSTER_TEXTURE_FLOAT
        clusterLightData.halfWidth = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_AREA_DATA_WIDTH).xyz;
        clusterLightData.halfHeight = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_AREA_DATA_HEIGHT).xyz;
    #else
        vec4 areaWidthX = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_WIDTH_X);
        vec4 areaWidthY = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_WIDTH_Y);
        vec4 areaWidthZ = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_WIDTH_Z);
        clusterLightData.halfWidth = vec3(mantissaExponent2Float(areaWidthX), mantissaExponent2Float(areaWidthY), mantissaExponent2Float(areaWidthZ));

        vec4 areaHeightX = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_HEIGHT_X);
        vec4 areaHeightY = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_HEIGHT_Y);
        vec4 areaHeightZ = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_AREA_DATA_HEIGHT_Z);
        clusterLightData.halfHeight = vec3(mantissaExponent2Float(areaHeightX), mantissaExponent2Float(areaHeightY), mantissaExponent2Float(areaHeightZ));
    #endif
}

void decodeClusterLightProjectionMatrixData(inout ClusterLightData clusterLightData) {
    
    // shadow matrix
    #ifdef CLUSTER_TEXTURE_FLOAT
        vec4 m0 = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_PROJ_MAT_0);
        vec4 m1 = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_PROJ_MAT_1);
        vec4 m2 = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_PROJ_MAT_2);
        vec4 m3 = sampleLightTextureF(clusterLightData, CLUSTER_TEXTURE_F_PROJ_MAT_3);
    #else
        vec4 m00 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_00);
        vec4 m01 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_01);
        vec4 m02 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_02);
        vec4 m03 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_03);
        vec4 m0 = decodeClusterLowRange4Vec4(m00, m01, m02, m03);

        vec4 m10 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_10);
        vec4 m11 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_11);
        vec4 m12 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_12);
        vec4 m13 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_13);
        vec4 m1 = decodeClusterLowRange4Vec4(m10, m11, m12, m13);

        vec4 m20 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_20);
        vec4 m21 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_21);
        vec4 m22 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_22);
        vec4 m23 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_23);
        vec4 m2 = decodeClusterLowRange4Vec4(m20, m21, m22, m23);

        vec4 m30 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_30);
        vec4 m31 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_31);
        vec4 m32 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_32);
        vec4 m33 = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_PROJ_MAT_33);
        vec4 m3 = vec4(mantissaExponent2Float(m30), mantissaExponent2Float(m31), mantissaExponent2Float(m32), mantissaExponent2Float(m33));
    #endif
    
    lightProjectionMatrix = mat4(m0, m1, m2, m3);
}

void decodeClusterLightShadowData(inout ClusterLightData clusterLightData) {
    
    // shadow biases
    vec4 biases = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_SHADOW_BIAS);
    clusterLightData.shadowBias = bytes2floatRange2(biases.xy, -1.0, 20.0),
    clusterLightData.shadowNormalBias = bytes2float2(biases.zw);
}

void decodeClusterLightCookieData(inout ClusterLightData clusterLightData) {

    vec4 cookieA = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_COOKIE_A);
    clusterLightData.cookieIntensity = cookieA.x;
    clusterLightData.cookieRgb = cookieA.y;

    clusterLightData.cookieChannelMask = sampleLightsTexture8(clusterLightData, CLUSTER_TEXTURE_8_COOKIE_B);
}

void evaluateLight(ClusterLightData light) {

    dAtten3 = vec3(1.0);

    // evaluate omni part of the light
    getLightDirPoint(light.position);

    #ifdef CLUSTER_AREALIGHTS

    // distance attenuation
    if (isClusteredLightArea(light)) { // area light

        // area lights
        decodeClusterLightAreaData(light);

        // handle light shape
        if (isClusteredLightRect(light)) {
            calcRectLightValues(light.position, light.halfWidth, light.halfHeight);
        } else if (isClusteredLightDisk(light)) {
            calcDiskLightValues(light.position, light.halfWidth, light.halfHeight);
        } else { // sphere
            calcSphereLightValues(light.position, light.halfWidth, light.halfHeight);
        }

        dAtten = getFalloffWindow(light.range);

    } else

    #endif

    {   // punctual light

        if (isClusteredLightFalloffLinear(light))
            dAtten = getFalloffLinear(light.range);
        else
            dAtten = getFalloffInvSquared(light.range);
    }

    if (dAtten > 0.00001) {

        #ifdef CLUSTER_AREALIGHTS

        if (isClusteredLightArea(light)) { // area light

            // handle light shape
            if (isClusteredLightRect(light)) {
                dAttenD = getRectLightDiffuse() * 16.0;
            } else if (isClusteredLightDisk(light)) {
                dAttenD = getDiskLightDiffuse() * 16.0;
            } else { // sphere
                dAttenD = getSphereLightDiffuse() * 16.0;
            }

        } else

        #endif

        {
            dAtten *= getLightDiffuse();
        }

        // spot light falloff
        if (isClusteredLightSpot(light)) {
            decodeClusterLightSpot(light);
            dAtten *= getSpotEffect(light.direction, light.innerConeAngleCos, light.outerConeAngleCos);
        }

        #if defined(CLUSTER_COOKIES_OR_SHADOWS)

        if (dAtten > 0.00001) {

            // shadow / cookie
            if (isClusteredLightCastShadow(light) || isClusteredLightCookie(light)) {

                // shared shadow / cookie data depends on light type
                if (isClusteredLightSpot(light)) {
                    decodeClusterLightProjectionMatrixData(light);
                } else {
                    decodeClusterLightOmniAtlasViewport(light);
                }

                float shadowTextureResolution = shadowAtlasParams.x;
                float shadowEdgePixels = shadowAtlasParams.y;

                #ifdef CLUSTER_COOKIES

                // cookie
                if (isClusteredLightCookie(light)) {
                    decodeClusterLightCookieData(light);

                    if (isClusteredLightSpot(light)) {
                        dAtten3 = getCookie2DClustered(TEXTURE_PASS(cookieAtlasTexture), lightProjectionMatrix, vPositionW, light.cookieIntensity, isClusteredLightCookieRgb(light), light.cookieChannelMask);
                    } else {
                        dAtten3 = getCookieCubeClustered(TEXTURE_PASS(cookieAtlasTexture), dLightDirW, light.cookieIntensity, isClusteredLightCookieRgb(light), light.cookieChannelMask, shadowTextureResolution, shadowEdgePixels, light.omniAtlasViewport);
                    }
                }

                #endif

                #ifdef CLUSTER_SHADOWS

                // shadow
                if (isClusteredLightCastShadow(light)) {
                    decodeClusterLightShadowData(light);

                    vec4 shadowParams = vec4(shadowTextureResolution, light.shadowNormalBias, light.shadowBias, 1.0 / light.range);

                    if (isClusteredLightSpot(light)) {

                        // spot shadow
                        getShadowCoordPerspZbufferNormalOffset(lightProjectionMatrix, shadowParams);
                        
                        #if defined(CLUSTER_SHADOW_TYPE_PCF1)
                            float shadow = getShadowSpotClusteredPCF1(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams);
                        #elif defined(CLUSTER_SHADOW_TYPE_PCF3)
                            float shadow = getShadowSpotClusteredPCF3(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams);
                        #elif defined(CLUSTER_SHADOW_TYPE_PCF5)
                            float shadow = getShadowSpotClusteredPCF5(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams);
                        #endif
                        dAtten *= mix(1.0, shadow, light.shadowIntensity);

                    } else {

                        // omni shadow
                        normalOffsetPointShadow(shadowParams);  // normalBias adjusted for distance

                        #if defined(CLUSTER_SHADOW_TYPE_PCF1)
                            float shadow = getShadowOmniClusteredPCF1(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams, light.omniAtlasViewport, shadowEdgePixels, dLightDirW);
                        #elif defined(CLUSTER_SHADOW_TYPE_PCF3)
                            float shadow = getShadowOmniClusteredPCF3(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams, light.omniAtlasViewport, shadowEdgePixels, dLightDirW);
                        #elif defined(CLUSTER_SHADOW_TYPE_PCF5)
                            float shadow = getShadowOmniClusteredPCF5(SHADOWMAP_PASS(shadowAtlasTexture), shadowParams, light.omniAtlasViewport, shadowEdgePixels, dLightDirW);
                        #endif
                        dAtten *= mix(1.0, shadow, light.shadowIntensity);
                    }
                }

                #endif
            }
        }

        #endif

        // diffuse / specular / clearcoat
        #ifdef CLUSTER_AREALIGHTS

        if (isClusteredLightArea(light)) { // area light

            // area light diffuse
            {
                vec3 areaDiffuse = (dAttenD * dAtten) * light.color * dAtten3;

                #if defined(LIT_SPECULAR)
                    #if defined(LIT_CONSERVE_ENERGY)
                        areaDiffuse = mix(areaDiffuse, vec3(0), dLTCSpecFres);
                    #endif
                #endif

                // area light diffuse - it does not mix diffuse lighting into specular attenuation
                dDiffuseLight += areaDiffuse;
            }

            // specular and clear coat are material settings and get included by a define based on the material
            #ifdef LIT_SPECULAR

                // area light specular
                float areaLightSpecular;

                if (isClusteredLightRect(light)) {
                    areaLightSpecular = getRectLightSpecular();
                } else if (isClusteredLightDisk(light)) {
                    areaLightSpecular = getDiskLightSpecular();
                } else { // sphere
                    areaLightSpecular = getSphereLightSpecular();
                }

                dSpecularLight += dLTCSpecFres * areaLightSpecular * dAtten * light.color * dAtten3;

                #ifdef LIT_CLEARCOAT

                    // area light specular clear coat
                    float areaLightSpecularCC;

                    if (isClusteredLightRect(light)) {
                        areaLightSpecularCC = getRectLightSpecularCC();
                    } else if (isClusteredLightDisk(light)) {
                        areaLightSpecularCC = getDiskLightSpecularCC();
                    } else { // sphere
                        areaLightSpecularCC = getSphereLightSpecularCC();
                    }

                    ccSpecularLight += ccLTCSpecFres * areaLightSpecularCC * dAtten * light.color  * dAtten3;

                #endif

            #endif

        } else

        #endif

        {    // punctual light

            // punctual light diffuse
            {
                vec3 punctualDiffuse = dAtten * light.color * dAtten3;

                #if defined(CLUSTER_AREALIGHTS)
                #if defined(LIT_SPECULAR)
                #if defined(LIT_CONSERVE_ENERGY)
                    punctualDiffuse = mix(punctualDiffuse, vec3(0), dSpecularity);
                #endif
                #endif
                #endif

                dDiffuseLight += punctualDiffuse;
            }
   
            // specular and clear coat are material settings and get included by a define based on the material
            #ifdef LIT_SPECULAR

                vec3 halfDir = normalize(-dLightDirNormW + dViewDirW);
                
                // specular
                #ifdef LIT_SPECULAR_FRESNEL
                    dSpecularLight += getLightSpecular(halfDir) * dAtten * light.color * dAtten3 * getFresnel(dot(dViewDirW, halfDir), dSpecularity);
                #else
                    dSpecularLight += getLightSpecular(halfDir) * dAtten * light.color * dAtten3 * dSpecularity;
                #endif

                #ifdef LIT_CLEARCOAT
                    #ifdef LIT_SPECULAR_FRESNEL
                        ccSpecularLight += getLightSpecularCC(halfDir) * dAtten * light.color * dAtten3 * getFresnelCC(dot(dViewDirW, halfDir));
                    #else
                        ccSpecularLight += getLightSpecularCC(halfDir) * dAtten * light.color * dAtten3;
                    #endif
                #endif

                #ifdef LIT_SHEEN
                    sSpecularLight += getLightSpecularSheen(halfDir) * dAtten * light.color * dAtten3;
                #endif

            #endif
        }
    }
}

void evaluateClusterLight(float lightIndex) {

    // decode core light data from textures
    ClusterLightData clusterLightData;
    decodeClusterLightCore(clusterLightData, lightIndex);

    // evaluate light if it uses accepted light mask
    if (acceptLightMask(clusterLightData))
        evaluateLight(clusterLightData);
}

void addClusteredLights() {
    // world space position to 3d integer cell cordinates in the cluster structure
    vec3 cellCoords = floor((vPositionW - clusterBoundsMin) * clusterCellsCountByBoundsSize);

    // no lighting when cell coordinate is out of range
    if (!(any(lessThan(cellCoords, vec3(0.0))) || any(greaterThanEqual(cellCoords, clusterCellsMax)))) {

        // cell index (mapping from 3d cell coordinates to linear memory)
        float cellIndex = dot(clusterCellsDot, cellCoords);

        // convert cell index to uv coordinates
        float clusterV = floor(cellIndex * clusterTextureSize.y);
        float clusterU = cellIndex - (clusterV * clusterTextureSize.x);

        #ifdef GL2

            // loop over maximum number of light cells
            for (int lightCellIndex = 0; lightCellIndex < clusterMaxCells; lightCellIndex++) {

                // using a single channel texture with data in alpha channel
                float lightIndex = texelFetch(clusterWorldTexture, ivec2(int(clusterU) + lightCellIndex, clusterV), 0).x;

                if (lightIndex <= 0.0)
                        return;

                evaluateClusterLight(lightIndex * 255.0); 
            }

        #else

            clusterV = (clusterV + 0.5) * clusterTextureSize.z;

            // loop over maximum possible number of supported light cells
            const float maxLightCells = 256.0;
            for (float lightCellIndex = 0.5; lightCellIndex < maxLightCells; lightCellIndex++) {

                float lightIndex = texture2DLodEXT(clusterWorldTexture, vec2(clusterTextureSize.y * (clusterU + lightCellIndex), clusterV), 0.0).x;

                if (lightIndex <= 0.0)
                    return;
                
                evaluateClusterLight(lightIndex * 255.0); 

                // end of the cell array
                if (lightCellIndex >= clusterMaxCells) {
                    break;
                }
            }

        #endif
    }
}
```
engine\src\scene\shader-lib\chunks\lit\frag\clusteredLightCookies.js
```glsl
vec3 _getCookieClustered(TEXTURE_ACCEPT(tex), vec2 uv, float intensity, bool isRgb, vec4 cookieChannel) {
    vec4 pixel = mix(vec4(1.0), texture2DLodEXT(tex, uv, 0.0), intensity);
    return isRgb == true ? pixel.rgb : vec3(dot(pixel, cookieChannel));
}

// getCookie2D for clustered lighting including channel selector
vec3 getCookie2DClustered(TEXTURE_ACCEPT(tex), mat4 transform, vec3 worldPosition, float intensity, bool isRgb, vec4 cookieChannel) {
    vec4 projPos = transform * vec4(worldPosition, 1.0);
    return _getCookieClustered(TEXTURE_PASS(tex), projPos.xy / projPos.w, intensity, isRgb, cookieChannel);
}

// getCookie for clustered omni light with the cookie texture being stored in the cookie atlas
vec3 getCookieCubeClustered(TEXTURE_ACCEPT(tex), vec3 dir, float intensity, bool isRgb, vec4 cookieChannel, float shadowTextureResolution, float shadowEdgePixels, vec3 omniAtlasViewport) {
    vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);
    return _getCookieClustered(TEXTURE_PASS(tex), uv, intensity, isRgb, cookieChannel);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\clusteredLightShadows.js
```glsl
// Clustered Omni Sampling using atlas

#ifdef GL2

    #if defined(CLUSTER_SHADOW_TYPE_PCF1)

    float getShadowOmniClusteredPCF1(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        return textureShadow(shadowMap, vec3(uv, shadowZ));
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF3)

    float getShadowOmniClusteredPCF3(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        dShadowCoord = vec3(uv, shadowZ);
        return getShadowPCF3x3(SHADOWMAP_PASS(shadowMap), shadowParams.xyz);
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF5)

    float getShadowOmniClusteredPCF5(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        dShadowCoord = vec3(uv, shadowZ);
        return getShadowPCF5x5(SHADOWMAP_PASS(shadowMap), shadowParams.xyz);
    }

    #endif

#else

    #if defined(CLUSTER_SHADOW_TYPE_PCF1)

    float getShadowOmniClusteredPCF1(sampler2D shadowMap, vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        // no filter shadow sampling
        float depth = unpackFloat(textureShadow(shadowMap, uv));
        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        return depth > shadowZ ? 1.0 : 0.0;
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF3)

    float getShadowOmniClusteredPCF3(sampler2D shadowMap, vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        // pcf3
        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        dShadowCoord = vec3(uv, shadowZ);
        return getShadowPCF3x3(shadowMap, shadowParams.xyz);
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF5)

    // we don't have PCF5 implementation for webgl1, use PCF3
    float getShadowOmniClusteredPCF5(sampler2D shadowMap, vec4 shadowParams, vec3 omniAtlasViewport, float shadowEdgePixels, vec3 dir) {

        float shadowTextureResolution = shadowParams.x;
        vec2 uv = getCubemapAtlasCoordinates(omniAtlasViewport, shadowEdgePixels, shadowTextureResolution, dir);

        // pcf3
        float shadowZ = length(dir) * shadowParams.w + shadowParams.z;
        dShadowCoord = vec3(uv, shadowZ);
        return getShadowPCF3x3(shadowMap, shadowParams.xyz);
    }

    #endif

#endif


// Clustered Spot Sampling using atlas

#ifdef GL2

    #if defined(CLUSTER_SHADOW_TYPE_PCF1)

    float getShadowSpotClusteredPCF1(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams) {
        return textureShadow(shadowMap, dShadowCoord);
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF3)

    float getShadowSpotClusteredPCF3(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams) {
        return getShadowSpotPCF3x3(SHADOWMAP_PASS(shadowMap), shadowParams);
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF5)

    float getShadowSpotClusteredPCF5(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams) {
        return getShadowPCF5x5(SHADOWMAP_PASS(shadowMap), shadowParams.xyz);
    }
    #endif

#else

    #if defined(CLUSTER_SHADOW_TYPE_PCF1)

    float getShadowSpotClusteredPCF1(sampler2D shadowMap, vec4 shadowParams) {

        float depth = unpackFloat(textureShadow(shadowMap, dShadowCoord.xy));

        return depth > dShadowCoord.z ? 1.0 : 0.0;

    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF3)

    float getShadowSpotClusteredPCF3(sampler2D shadowMap, vec4 shadowParams) {
        return getShadowSpotPCF3x3(shadowMap, shadowParams);
    }

    #endif

    #if defined(CLUSTER_SHADOW_TYPE_PCF5)

    // we don't have PCF5 implementation for webgl1, use PCF3
    float getShadowSpotClusteredPCF5(sampler2D shadowMap, vec4 shadowParams) {
        return getShadowSpotPCF3x3(shadowMap, shadowParams);
    }

    #endif

#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\clusteredLightUtils.js
```glsl
// Converts unnormalized direction vector to a cubemap face index [0..5] and uv coordinates within the face in [0..1] range.
// Additionally offset to a tile in atlas within 3x3 subdivision is provided
vec2 getCubemapFaceCoordinates(const vec3 dir, out float faceIndex, out vec2 tileOffset)
{
    vec3 vAbs = abs(dir);
    float ma;
    vec2 uv;
    if (vAbs.z >= vAbs.x && vAbs.z >= vAbs.y) {   // front / back

        faceIndex = dir.z < 0.0 ? 5.0 : 4.0;
        ma = 0.5 / vAbs.z;
        uv = vec2(dir.z < 0.0 ? -dir.x : dir.x, -dir.y);

        tileOffset.x = 2.0;
        tileOffset.y = dir.z < 0.0 ? 1.0 : 0.0;

    } else if(vAbs.y >= vAbs.x) {  // top index 2, bottom index 3

        faceIndex = dir.y < 0.0 ? 3.0 : 2.0;
        ma = 0.5 / vAbs.y;
        uv = vec2(dir.x, dir.y < 0.0 ? -dir.z : dir.z);

        tileOffset.x = 1.0;
        tileOffset.y = dir.y < 0.0 ? 1.0 : 0.0;

    } else {    // left / right

        faceIndex = dir.x < 0.0 ? 1.0 : 0.0;
        ma = 0.5 / vAbs.x;
        uv = vec2(dir.x < 0.0 ? dir.z : -dir.z, -dir.y);

        tileOffset.x = 0.0;
        tileOffset.y = dir.x < 0.0 ? 1.0 : 0.0;

    }
    return uv * ma + 0.5;
}

// converts unnormalized direction vector to a texture coordinate for a cubemap face stored within texture atlas described by the viewport
vec2 getCubemapAtlasCoordinates(const vec3 omniAtlasViewport, float shadowEdgePixels, float shadowTextureResolution, const vec3 dir) {

    float faceIndex;
    vec2 tileOffset;
    vec2 uv = getCubemapFaceCoordinates(dir, faceIndex, tileOffset);

    // move uv coordinates inwards inside to compensate for larger fov when rendering shadow into atlas
    float atlasFaceSize = omniAtlasViewport.z;
    float tileSize = shadowTextureResolution * atlasFaceSize;
    float offset = shadowEdgePixels / tileSize;
    uv = uv * vec2(1.0 - offset * 2.0) + vec2(offset * 1.0);

    // scale uv coordinates to cube face area within the viewport
    uv *= atlasFaceSize;

    // offset into face of the atlas (3x3 grid)
    uv += tileOffset * atlasFaceSize;

    // offset into the atlas viewport
    uv += omniAtlasViewport.xy;

    return uv;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\combine.js
```glsl
vec3 combineColor() {
    vec3 ret = vec3(0);
#ifdef LIT_OLD_AMBIENT
    ret += (dDiffuseLight - light_globalAmbient) * dAlbedo + material_ambient * light_globalAmbient;
#else
    ret += dAlbedo * dDiffuseLight;
#endif
#ifdef LIT_SPECULAR
    ret += dSpecularLight;
#endif
#ifdef LIT_REFLECTIONS
    ret += dReflection.rgb * dReflection.a;
#endif

#ifdef LIT_SHEEN
    float sheenScaling = 1.0 - max(max(sSpecularity.r, sSpecularity.g), sSpecularity.b) * 0.157;
    ret = ret * sheenScaling + (sSpecularLight + sReflection.rgb) * sSpecularity;
#endif
#ifdef LIT_CLEARCOAT
    float clearCoatScaling = 1.0 - ccFresnel * ccSpecularity;
    ret = ret * clearCoatScaling + (ccSpecularLight + ccReflection.rgb) * ccSpecularity;
#endif

    return ret;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\cookie.js
```glsl
// light cookie functionality for non-clustered lights
vec4 getCookie2D(sampler2D tex, mat4 transform, float intensity) {
    vec4 projPos = transform * vec4(vPositionW, 1.0);
    projPos.xy /= projPos.w;
    return mix(vec4(1.0), texture2D(tex, projPos.xy), intensity);
}

vec4 getCookie2DClip(sampler2D tex, mat4 transform, float intensity) {
    vec4 projPos = transform * vec4(vPositionW, 1.0);
    projPos.xy /= projPos.w;
    if (projPos.x < 0.0 || projPos.x > 1.0 || projPos.y < 0.0 || projPos.y > 1.0 || projPos.z < 0.0) return vec4(0.0);
    return mix(vec4(1.0), texture2D(tex, projPos.xy), intensity);
}

vec4 getCookie2DXform(sampler2D tex, mat4 transform, float intensity, vec4 cookieMatrix, vec2 cookieOffset) {
    vec4 projPos = transform * vec4(vPositionW, 1.0);
    projPos.xy /= projPos.w;
    projPos.xy += cookieOffset;
    vec2 uv = mat2(cookieMatrix) * (projPos.xy-vec2(0.5)) + vec2(0.5);
    return mix(vec4(1.0), texture2D(tex, uv), intensity);
}

vec4 getCookie2DClipXform(sampler2D tex, mat4 transform, float intensity, vec4 cookieMatrix, vec2 cookieOffset) {
    vec4 projPos = transform * vec4(vPositionW, 1.0);
    projPos.xy /= projPos.w;
    projPos.xy += cookieOffset;
    if (projPos.x < 0.0 || projPos.x > 1.0 || projPos.y < 0.0 || projPos.y > 1.0 || projPos.z < 0.0) return vec4(0.0);
    vec2 uv = mat2(cookieMatrix) * (projPos.xy-vec2(0.5)) + vec2(0.5);
    return mix(vec4(1.0), texture2D(tex, uv), intensity);
}

vec4 getCookieCube(samplerCube tex, mat4 transform, float intensity) {
    return mix(vec4(1.0), textureCube(tex, dLightDirNormW * mat3(transform)), intensity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\cubeMapProjectBox.js
```glsl
uniform vec3 envBoxMin;
uniform vec3 envBoxMax;

vec3 cubeMapProject(vec3 nrdir) {
    nrdir = cubeMapRotate(nrdir);

    vec3 rbmax = (envBoxMax - vPositionW) / nrdir;
    vec3 rbmin = (envBoxMin - vPositionW) / nrdir;

    vec3 rbminmax;
    rbminmax.x = nrdir.x>0.0? rbmax.x : rbmin.x;
    rbminmax.y = nrdir.y>0.0? rbmax.y : rbmin.y;
    rbminmax.z = nrdir.z>0.0? rbmax.z : rbmin.z;

    float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);

    vec3 posonbox = vPositionW + nrdir * fa;
    vec3 envBoxPos = (envBoxMin + envBoxMax) * 0.5;
    return normalize(posonbox - envBoxPos);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\cubeMapProjectNone.js
```glsl
vec3 cubeMapProject(vec3 dir) {
    return cubeMapRotate(dir);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\cubeMapRotate.js
```glsl
#ifdef CUBEMAP_ROTATION
uniform mat3 cubeMapRotationMatrix;
#endif

vec3 cubeMapRotate(vec3 refDir) {
#ifdef CUBEMAP_ROTATION
    return refDir * cubeMapRotationMatrix;
#else
    return refDir;
#endif
}
```
engine\src\scene\shader-lib\chunks\lit\frag\end.js
```glsl
    gl_FragColor.rgb = combineColor();

    gl_FragColor.rgb += dEmission;
    gl_FragColor.rgb = addFog(gl_FragColor.rgb);

    #ifndef HDR
    gl_FragColor.rgb = toneMap(gl_FragColor.rgb);
    gl_FragColor.rgb = gammaCorrectOutput(gl_FragColor.rgb);
    #endif
```
engine\src\scene\shader-lib\chunks\lit\frag\falloffInvSquared.js
```glsl
float getFalloffWindow(float lightRadius) {
    float sqrDist = dot(dLightDirW, dLightDirW);
    float invRadius = 1.0 / lightRadius;
    return square( saturate( 1.0 - square( sqrDist * square(invRadius) ) ) );
}

float getFalloffInvSquared(float lightRadius) {
    float sqrDist = dot(dLightDirW, dLightDirW);
    float falloff = 1.0 / (sqrDist + 1.0);
    float invRadius = 1.0 / lightRadius;

    falloff *= 16.0;
    falloff *= square( saturate( 1.0 - square( sqrDist * square(invRadius) ) ) );

    return falloff;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\falloffLinear.js
```glsl
float getFalloffLinear(float lightRadius) {
    float d = length(dLightDirW);
    return max(((lightRadius - d) / lightRadius), 0.0);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\float-unpacking.js
```glsl
// float unpacking functionality, complimentary to float-packing.js
float bytes2float2(vec2 data) {
    return dot(data, vec2(1.0, 1.0 / 255.0));
}

float bytes2float3(vec3 data) {
    return dot(data, vec3(1.0, 1.0 / 255.0, 1.0 / 65025.0));
}

float bytes2float4(vec4 data) {
    return dot(data, vec4(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
}

float bytes2floatRange2(vec2 data, float min, float max) {
    return mix(min, max, bytes2float2(data));
}

float bytes2floatRange3(vec3 data, float min, float max) {
    return mix(min, max, bytes2float3(data));
}

float bytes2floatRange4(vec4 data, float min, float max) {
    return mix(min, max, bytes2float4(data));
}

float mantissaExponent2Float(vec4 pack)
{
    float value = bytes2floatRange3(pack.xyz, -1.0, 1.0);
    float exponent = floor(pack.w * 255.0 - 127.0);
    return value * exp2(exponent);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\fogExp.js
```glsl
uniform vec3 fog_color;
uniform float fog_density;
float dBlendModeFogFactor = 1.0;

vec3 addFog(vec3 color) {
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = exp(-depth * fog_density);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    return mix(fog_color * dBlendModeFogFactor, color, fogFactor);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\fogExp2.js
```glsl
uniform vec3 fog_color;
uniform float fog_density;
float dBlendModeFogFactor = 1.0;

vec3 addFog(vec3 color) {
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = exp(-depth * depth * fog_density * fog_density);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    return mix(fog_color * dBlendModeFogFactor, color, fogFactor);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\fogLinear.js
```glsl
uniform vec3 fog_color;
uniform float fog_start;
uniform float fog_end;
float dBlendModeFogFactor = 1.0;

vec3 addFog(vec3 color) {
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = (fog_end - depth) / (fog_end - fog_start);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    return mix(fog_color * dBlendModeFogFactor, color, fogFactor);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\fogNone.js
```glsl
float dBlendModeFogFactor = 1.0;

vec3 addFog(vec3 color) {
    return color;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\fresnelSchlick.js
```glsl
// Schlick's approximation
vec3 getFresnel(float cosTheta, vec3 f0) {
    float fresnel = pow(1.0 - max(cosTheta, 0.0), 5.0);
    float glossSq = dGlossiness * dGlossiness;
    vec3 ret = f0 + (max(vec3(glossSq), f0) - f0) * fresnel;
    #ifdef LIT_IRIDESCENCE
        return mix(ret, dIridescenceFresnel, vec3(dIridescence));
    #else
        return ret;
    #endif    
}

float getFresnelCC(float cosTheta) {
    float fresnel = pow(1.0 - max(cosTheta, 0.0), 5.0);
    return 0.04 + (1.0 - 0.04) * fresnel;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\iridescenceDiffraction.js
```glsl
uniform float material_iridescenceRefractionIndex;

#ifndef PI
#define PI 3.14159265
#endif

float iridescence_iorToFresnel(float transmittedIor, float incidentIor) {
    return pow((transmittedIor - incidentIor) / (transmittedIor + incidentIor), 2.0);
}

vec3 iridescence_iorToFresnel(vec3 transmittedIor, float incidentIor) {
    return pow((transmittedIor - vec3(incidentIor)) / (transmittedIor + vec3(incidentIor)), vec3(2.0));
}

vec3 iridescence_fresnelToIor(vec3 f0) {
    vec3 sqrtF0 = sqrt(f0);
    return (vec3(1.0) + sqrtF0) / (vec3(1.0) - sqrtF0);
}

vec3 iridescence_sensitivity(float opd, vec3 shift) {
    float phase = 2.0 * PI * opd * 1.0e-9;
    const vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    const vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    const vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    vec3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-pow(phase, 2.0) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * pow(phase, 2.0));
    xyz /= vec3(1.0685e-07);

    const mat3 XYZ_TO_REC709 = mat3(
        3.2404542, -0.9692660,  0.0556434,
       -1.5371385,  1.8760108, -0.2040259,
       -0.4985314,  0.0415560,  1.0572252
    );

    return XYZ_TO_REC709 * xyz;
}

float iridescence_fresnel(float cosTheta, float f0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return f0 + (1.0 - f0) * x5;
} 

vec3 iridescence_fresnel(float cosTheta, vec3 f0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    float x5 = x * x2 * x2; 
    return f0 + (vec3(1.0) - f0) * x5;
}

vec3 calcIridescence(float outsideIor, float cosTheta, vec3 base_f0) {

    float iridescenceIor = mix(outsideIor, material_iridescenceRefractionIndex, smoothstep(0.0, 0.03, dIridescenceThickness));
    float sinTheta2Sq = pow(outsideIor / iridescenceIor, 2.0) * (1.0 - pow(cosTheta, 2.0));
    float cosTheta2Sq = 1.0 - sinTheta2Sq;

    if (cosTheta2Sq < 0.0) {
        return vec3(1.0);
    }

    float cosTheta2 = sqrt(cosTheta2Sq);

    float r0 = iridescence_iorToFresnel(iridescenceIor, outsideIor);
    float r12 = iridescence_fresnel(cosTheta, r0);
    float r21 = r12;
    float t121 = 1.0 - r12;

    float phi12 = iridescenceIor < outsideIor ? PI : 0.0;
    float phi21 = PI - phi12;

    vec3 baseIor = iridescence_fresnelToIor(base_f0 + vec3(0.0001));
    vec3 r1 = iridescence_iorToFresnel(baseIor, iridescenceIor);
    vec3 r23 = iridescence_fresnel(cosTheta2, r1);

    vec3 phi23 = vec3(0.0);
    if (baseIor[0] < iridescenceIor) phi23[0] = PI;
    if (baseIor[1] < iridescenceIor) phi23[1] = PI;
    if (baseIor[2] < iridescenceIor) phi23[2] = PI;
    float opd = 2.0 * iridescenceIor * dIridescenceThickness * cosTheta2;
    vec3 phi = vec3(phi21) + phi23; 

    vec3 r123Sq = clamp(r12 * r23, 1e-5, 0.9999);
    vec3 r123 = sqrt(r123Sq);
    vec3 rs = pow(t121, 2.0) * r23 / (1.0 - r123Sq);

    vec3 c0 = r12 + rs;
    vec3 i = c0;

    vec3 cm = rs - t121;
    for (int m = 1; m <= 2; m++) {
        cm *= r123;
        vec3 sm = 2.0 * iridescence_sensitivity(float(m) * opd, float(m) * phi);
        i += cm * sm;
    }
    return max(i, vec3(0.0));
}

void getIridescence(float cosTheta) {
    dIridescenceFresnel = calcIridescence(1.0, cosTheta, dSpecularity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightDiffuseLambert.js
```glsl
float getLightDiffuse() {
    return max(dot(dNormalW, -dLightDirNormW), 0.0);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightDirPoint.js
```glsl
void getLightDirPoint(vec3 lightPosW) {
    dLightDirW = vPositionW - lightPosW;
    dLightDirNormW = normalize(dLightDirW);
    dLightPosW = lightPosW;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightmapAdd.js
```glsl
void addLightMap() {
    dDiffuseLight += dLightmap;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightmapDirAdd.js
```glsl
void addLightMap() {
    if (dot(dLightmapDir, dLightmapDir) < 0.0001) {
        dDiffuseLight += dLightmap;
    } else {
        dLightDirNormW = dLightmapDir;

        float vlight = saturate(dot(dLightDirNormW, -dVertexNormalW));
        float flight = saturate(dot(dLightDirNormW, -dNormalW));
        float nlight = (flight / max(vlight, 0.01)) * 0.5;

        dDiffuseLight += dLightmap * nlight * 2.0;

        vec3 halfDirW = normalize(-dLightmapDir + dViewDirW);
        vec3 specularLight = dLightmap * getLightSpecular(halfDirW);

        #ifdef LIT_SPECULAR_FRESNEL
        specularLight *= getFresnel(dot(dViewDirW, halfDirW), dSpecularity);
        #endif

        dSpecularLight += specularLight;
    }
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightSheen.js
```glsl

float sheenD(vec3 normal, vec3 h, float roughness) {
    float invR = 1.0 / (roughness * roughness);
    float cos2h = max(dot(normal, h), 0.0);
    cos2h *= cos2h;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

float sheenV(vec3 normal, vec3 view, vec3 light) {
    float NoV = max(dot(normal, view), 0.000001);
    float NoL = max(dot(normal, light), 0.000001);
    return 1.0 / (4.0 * (NoL + NoV - NoL * NoV));
}

float getLightSpecularSheen(vec3 h) {
    float D = sheenD(dNormalW, h, sGlossiness);
    float V = sheenV(dNormalW, dViewDirW, -dLightDirNormW);
    return D * V;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\lightSpecularAnisoGGX.js
```glsl
// Anisotropic GGX
float calcLightSpecular(float tGlossiness, vec3 tNormalW, vec3 h) {
    float PI = 3.141592653589793;
    float roughness = max((1.0 - tGlossiness) * (1.0 - tGlossiness), 0.001);
    float anisotropy = material_anisotropy * roughness;
 
    float at = max((roughness + anisotropy), roughness / 4.0);
    float ab = max((roughness - anisotropy), roughness / 4.0);

    float NoH = dot(tNormalW, h);
    float ToH = dot(dTBN[0], h);
    float BoH = dot(dTBN[1], h);

    float a2 = at * ab;
    vec3 v = vec3(ab * ToH, at * BoH, a2 * NoH);
    float v2 = dot(v, v);
    float w2 = a2 / v2;
    float D = a2 * w2 * w2 * (1.0 / PI);

    float ToV = dot(dTBN[0], dViewDirW);
    float BoV = dot(dTBN[1], dViewDirW);
    float ToL = dot(dTBN[0], -dLightDirNormW);
    float BoL = dot(dTBN[1], -dLightDirNormW);
    float NoV = dot(tNormalW, dViewDirW);
    float NoL = dot(tNormalW, -dLightDirNormW);

    float lambdaV = NoL * length(vec3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(vec3(at * ToL, ab * BoL, NoL));
    float G = 0.5 / (lambdaV + lambdaL);

    return D * G;
}

float getLightSpecular(vec3 h) {
    return calcLightSpecular(dGlossiness, dNormalW, h);
}

#ifdef LIT_CLEARCOAT
float getLightSpecularCC(vec3 h) {
    return calcLightSpecular(ccGlossiness, ccNormalW, h);
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\lightSpecularBlinn.js
```glsl
// Energy-conserving (hopefully) Blinn-Phong
float calcLightSpecular(float tGlossiness, vec3 tNormalW, vec3 h) {
    float nh = max( dot( h, tNormalW ), 0.0 );

    float specPow = exp2(tGlossiness * 11.0); // glossiness is linear, power is not; 0 - 2048

    // Hack: On Mac OS X, calling pow with zero for the exponent generates hideous artifacts so bias up a little
    specPow = max(specPow, 0.0001);

    return pow(nh, specPow) * (specPow + 2.0) / 8.0;
}

float getLightSpecular(vec3 h) {
    return calcLightSpecular(dGlossiness, dNormalW, h);
}

#ifdef LIT_CLEARCOAT
float getLightSpecularCC(vec3 h) {
    return calcLightSpecular(ccGlossiness, ccNormalW, h);
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\lightSpecularPhong.js
```glsl
float calcLightSpecular(float tGlossiness, vec3 tReflDirW, vec3 h) {
    float specPow = tGlossiness;

    // Hack: On Mac OS X, calling pow with zero for the exponent generates hideous artifacts so bias up a little
    return pow(max(dot(tReflDirW, -dLightDirNormW), 0.0), specPow + 0.0001);
}

float getLightSpecular(vec3 h) {
    return calcLightSpecular(dGlossiness, dReflDirW, h);
}

#ifdef LIT_CLEARCOAT
float getLightSpecularCC(vec3 h) {
    return calcLightSpecular(ccGlossiness, ccReflDirW,h );
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\ltc.js
```glsl
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
// by Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
// code: https://github.com/selfshadow/ltc_code/

mat3 transposeMat3( const in mat3 m ) {
    mat3 tmp;
    tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
    tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
    tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
    return tmp;
}

vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
    const float LUT_SIZE = 64.0;
    const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
    const float LUT_BIAS = 0.5 / LUT_SIZE;
    float dotNV = saturate( dot( N, V ) );
    // texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
    vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
    uv = uv * LUT_SCALE + LUT_BIAS;
    return uv;
}

float LTC_ClippedSphereFormFactor( const in vec3 f ) {
    // Real-Time Area Lighting: a Journey from Research to Production (p.102)
    // An approximation of the form factor of a horizon-clipped rectangle.
    float l = length( f );
    return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}

vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
    float x = dot( v1, v2 );
    float y = abs( x );
    // rational polynomial approximation to theta / sin( theta ) / 2PI
    float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
    float b = 3.4175940 + ( 4.1616724 + y ) * y;
    float v = a / b;
    float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
    return cross( v1, v2 ) * theta_sintheta;
}

struct Coords {
    vec3 coord0;
    vec3 coord1;
    vec3 coord2;
    vec3 coord3;
};

float LTC_EvaluateRect( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in Coords rectCoords) {
    // bail if point is on back side of plane of light
    // assumes ccw winding order of light vertices
    vec3 v1 = rectCoords.coord1 - rectCoords.coord0;
    vec3 v2 = rectCoords.coord3 - rectCoords.coord0;
    
    vec3 lightNormal = cross( v1, v2 );
    // if( dot( lightNormal, P - rectCoords.coord0 ) < 0.0 ) return 0.0;
    float factor = sign(-dot( lightNormal, P - rectCoords.coord0 ));

    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize( V - N * dot( V, N ) );
    T2 =  factor * cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system
    // compute transform
    mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
    // transform rect
    vec3 coords[ 4 ];
    coords[ 0 ] = mat * ( rectCoords.coord0 - P );
    coords[ 1 ] = mat * ( rectCoords.coord1 - P );
    coords[ 2 ] = mat * ( rectCoords.coord2 - P );
    coords[ 3 ] = mat * ( rectCoords.coord3 - P );
    // project rect onto sphere
    coords[ 0 ] = normalize( coords[ 0 ] );
    coords[ 1 ] = normalize( coords[ 1 ] );
    coords[ 2 ] = normalize( coords[ 2 ] );
    coords[ 3 ] = normalize( coords[ 3 ] );
    // calculate vector form factor
    vec3 vectorFormFactor = vec3( 0.0 );
    vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
    vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
    vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
    vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
    // adjust for horizon clipping
    float result = LTC_ClippedSphereFormFactor( vectorFormFactor );

    return result;
}

Coords dLTCCoords;
Coords getLTCLightCoords(vec3 lightPos, vec3 halfWidth, vec3 halfHeight){
    Coords coords;
    coords.coord0 = lightPos + halfWidth - halfHeight;
    coords.coord1 = lightPos - halfWidth - halfHeight;
    coords.coord2 = lightPos - halfWidth + halfHeight;
    coords.coord3 = lightPos + halfWidth + halfHeight;
    return coords;
}

float dSphereRadius;
Coords getSphereLightCoords(vec3 lightPos, vec3 halfWidth, vec3 halfHeight){
    // used for simple sphere light falloff
    // also, the code only handles a spherical light, it cannot be non-uniformly scaled in world space, and so we enforce it here
    dSphereRadius = max(length(halfWidth), length(halfHeight));

    // Billboard the 2d light quad to reflection vector, as it's used for specular. This allows us to use disk math for the sphere.
    vec3 f = reflect(normalize(lightPos - view_position), vNormalW);
    vec3 w = normalize(cross(f, halfHeight));
    vec3 h = normalize(cross(f, w));

    return getLTCLightCoords(lightPos, w * dSphereRadius, h * dSphereRadius);
}

// used for LTC LUT texture lookup
vec2 dLTCUV;
#ifdef LIT_CLEARCOAT
vec2 ccLTCUV;
#endif
vec2 getLTCLightUV(float tGlossiness, vec3 tNormalW)
{
    float roughness = max((1.0 - tGlossiness) * (1.0 - tGlossiness), 0.001);
    return LTC_Uv( tNormalW, dViewDirW, roughness );
}

//used for energy conservation and to modulate specular
vec3 dLTCSpecFres;
#ifdef LIT_CLEARCOAT
vec3 ccLTCSpecFres;
#endif
vec3 getLTCLightSpecFres(vec2 uv, vec3 tSpecularity)
{
    vec4 t2 = texture2DLodEXT(areaLightsLutTex2, uv, 0.0);

    #ifdef AREA_R8_G8_B8_A8_LUTS
    t2 *= vec4(0.693103,1,1,1);
    t2 += vec4(0.306897,0,0,0);
    #endif

    return tSpecularity * t2.x + ( vec3( 1.0 ) - tSpecularity) * t2.y;
}

void calcLTCLightValues()
{
    dLTCUV = getLTCLightUV(dGlossiness, dNormalW);
    dLTCSpecFres = getLTCLightSpecFres(dLTCUV, dSpecularity); 

#ifdef LIT_CLEARCOAT
    ccLTCUV = getLTCLightUV(ccGlossiness, ccNormalW);
    ccLTCSpecFres = getLTCLightSpecFres(ccLTCUV, vec3(ccSpecularity));
#endif
}

void calcRectLightValues(vec3 lightPos, vec3 halfWidth, vec3 halfHeight)
{
    dLTCCoords = getLTCLightCoords(lightPos, halfWidth, halfHeight);
}
void calcDiskLightValues(vec3 lightPos, vec3 halfWidth, vec3 halfHeight)
{
    calcRectLightValues(lightPos, halfWidth, halfHeight);
}
void calcSphereLightValues(vec3 lightPos, vec3 halfWidth, vec3 halfHeight)
{
    dLTCCoords = getSphereLightCoords(lightPos, halfWidth, halfHeight);
}

// An extended version of the implementation from
// "How to solve a cubic equation, revisited"
// http://momentsingraphics.de/?p=105
vec3 SolveCubic(vec4 Coefficient)
{
    float pi = 3.14159;
    // Normalize the polynomial
    Coefficient.xyz /= Coefficient.w;
    // Divide middle coefficients by three
    Coefficient.yz /= 3.0;

    float A = Coefficient.w;
    float B = Coefficient.z;
    float C = Coefficient.y;
    float D = Coefficient.x;

    // Compute the Hessian and the discriminant
    vec3 Delta = vec3(
        -Coefficient.z * Coefficient.z + Coefficient.y,
        -Coefficient.y * Coefficient.z + Coefficient.x,
        dot(vec2(Coefficient.z, -Coefficient.y), Coefficient.xy)
    );

    float Discriminant = dot(vec2(4.0 * Delta.x, -Delta.y), Delta.zy);

    vec3 RootsA, RootsD;

    vec2 xlc, xsc;

    // Algorithm A
    {
        float A_a = 1.0;
        float C_a = Delta.x;
        float D_a = -2.0 * B * Delta.x + Delta.y;

        // Take the cubic root of a normalized complex number
        float Theta = atan(sqrt(Discriminant), -D_a) / 3.0;

        float x_1a = 2.0 * sqrt(-C_a) * cos(Theta);
        float x_3a = 2.0 * sqrt(-C_a) * cos(Theta + (2.0 / 3.0) * pi);

        float xl;
        if ((x_1a + x_3a) > 2.0 * B)
            xl = x_1a;
        else
            xl = x_3a;

        xlc = vec2(xl - B, A);
    }

    // Algorithm D
    {
        float A_d = D;
        float C_d = Delta.z;
        float D_d = -D * Delta.y + 2.0 * C * Delta.z;

        // Take the cubic root of a normalized complex number
        float Theta = atan(D * sqrt(Discriminant), -D_d) / 3.0;

        float x_1d = 2.0 * sqrt(-C_d) * cos(Theta);
        float x_3d = 2.0 * sqrt(-C_d) * cos(Theta + (2.0 / 3.0) * pi);

        float xs;
        if (x_1d + x_3d < 2.0 * C)
            xs = x_1d;
        else
            xs = x_3d;

        xsc = vec2(-D, xs + C);
    }

    float E =  xlc.y * xsc.y;
    float F = -xlc.x * xsc.y - xlc.y * xsc.x;
    float G =  xlc.x * xsc.x;

    vec2 xmc = vec2(C * F - B * G, -B * F + C * E);

    vec3 Root = vec3(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);

    if (Root.x < Root.y && Root.x < Root.z)
        Root.xyz = Root.yxz;
    else if (Root.z < Root.x && Root.z < Root.y)
        Root.xyz = Root.xzy;

    return Root;
}

float LTC_EvaluateDisk(vec3 N, vec3 V, vec3 P, mat3 Minv, Coords points)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N * dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    //mat3 R = transpose(mat3(T1, T2, N));
    mat3 R = transposeMat3( mat3( T1, T2, N ) );
    // polygon (allocate 5 vertices for clipping)
    vec3 L_[ 3 ];
    L_[ 0 ] = R * ( points.coord0 - P );
    L_[ 1 ] = R * ( points.coord1 - P );
    L_[ 2 ] = R * ( points.coord2 - P );

    vec3 Lo_i = vec3(0);

    // init ellipse
    vec3 C  = 0.5 * (L_[0] + L_[2]);
    vec3 V1 = 0.5 * (L_[1] - L_[2]);
    vec3 V2 = 0.5 * (L_[1] - L_[0]);

    C  = Minv * C;
    V1 = Minv * V1;
    V2 = Minv * V2;

    //if(dot(cross(V1, V2), C) > 0.0)
    //    return 0.0;

    // compute eigenvectors of ellipse
    float a, b;
    float d11 = dot(V1, V1);
    float d22 = dot(V2, V2);
    float d12 = dot(V1, V2);
    if (abs(d12) / sqrt(d11 * d22) > 0.0001)
    {
        float tr = d11 + d22;
        float det = -d12 * d12 + d11 * d22;

        // use sqrt matrix to solve for eigenvalues
        det = sqrt(det);
        float u = 0.5 * sqrt(tr - 2.0 * det);
        float v = 0.5 * sqrt(tr + 2.0 * det);
        float e_max = (u + v) * (u + v);
        float e_min = (u - v) * (u - v);

        vec3 V1_, V2_;

        if (d11 > d22)
        {
            V1_ = d12 * V1 + (e_max - d11) * V2;
            V2_ = d12 * V1 + (e_min - d11) * V2;
        }
        else
        {
            V1_ = d12*V2 + (e_max - d22)*V1;
            V2_ = d12*V2 + (e_min - d22)*V1;
        }

        a = 1.0 / e_max;
        b = 1.0 / e_min;
        V1 = normalize(V1_);
        V2 = normalize(V2_);
    }
    else
    {
        a = 1.0 / dot(V1, V1);
        b = 1.0 / dot(V2, V2);
        V1 *= sqrt(a);
        V2 *= sqrt(b);
    }

    vec3 V3 = cross(V1, V2);
    if (dot(C, V3) < 0.0)
        V3 *= -1.0;

    float L  = dot(V3, C);
    float x0 = dot(V1, C) / L;
    float y0 = dot(V2, C) / L;

    float E1 = inversesqrt(a);
    float E2 = inversesqrt(b);

    a *= L * L;
    b *= L * L;

    float c0 = a * b;
    float c1 = a * b * (1.0 + x0 * x0 + y0 * y0) - a - b;
    float c2 = 1.0 - a * (1.0 + x0 * x0) - b * (1.0 + y0 * y0);
    float c3 = 1.0;

    vec3 roots = SolveCubic(vec4(c0, c1, c2, c3));
    float e1 = roots.x;
    float e2 = roots.y;
    float e3 = roots.z;

    vec3 avgDir = vec3(a * x0 / (a - e2), b * y0 / (b - e2), 1.0);

    mat3 rotate = mat3(V1, V2, V3);

    avgDir = rotate * avgDir;
    avgDir = normalize(avgDir);

    float L1 = sqrt(-e2 / e3);
    float L2 = sqrt(-e2 / e1);

    float formFactor = L1 * L2 * inversesqrt((1.0 + L1 * L1) * (1.0 + L2 * L2));
    
    const float LUT_SIZE = 64.0;
    const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
    const float LUT_BIAS = 0.5 / LUT_SIZE;

    // use tabulated horizon-clipped sphere
    vec2 uv = vec2(avgDir.z * 0.5 + 0.5, formFactor);
    uv = uv*LUT_SCALE + LUT_BIAS;

    float scale = texture2DLodEXT(areaLightsLutTex2, uv, 0.0).w;

    return formFactor*scale;
}

float getRectLightDiffuse() {
    return LTC_EvaluateRect( dNormalW, dViewDirW, vPositionW, mat3( 1.0 ), dLTCCoords );
}

float getDiskLightDiffuse() {
    return LTC_EvaluateDisk( dNormalW, dViewDirW, vPositionW, mat3( 1.0 ), dLTCCoords );
}

float getSphereLightDiffuse() {
    // NB: this could be improved further with distance based wrap lighting
    float falloff = dSphereRadius / (dot(dLightDirW, dLightDirW) + dSphereRadius);
    return getLightDiffuse()*falloff;
}

mat3 getLTCLightInvMat(vec2 uv)
{
    vec4 t1 = texture2DLodEXT(areaLightsLutTex1, uv, 0.0);

    #ifdef AREA_R8_G8_B8_A8_LUTS
    t1 *= vec4(1.001, 0.3239, 0.60437568, 1.0);
    t1 += vec4(0.0, -0.2976, -0.01381, 0.0);
    #endif

    return mat3(
        vec3( t1.x, 0, t1.y ),
        vec3(    0, 1,    0 ),
        vec3( t1.z, 0, t1.w )
    );
}

float calcRectLightSpecular(vec3 tNormalW, vec2 uv) {
    mat3 mInv = getLTCLightInvMat(uv);
    return LTC_EvaluateRect( tNormalW, dViewDirW, vPositionW, mInv, dLTCCoords );
}

float getRectLightSpecular() {
    return calcRectLightSpecular(dNormalW, dLTCUV);
}

#ifdef LIT_CLEARCOAT
float getRectLightSpecularCC() {
    return calcRectLightSpecular(ccNormalW, ccLTCUV);
}
#endif

float calcDiskLightSpecular(vec3 tNormalW, vec2 uv) {
    mat3 mInv = getLTCLightInvMat(uv);
    return LTC_EvaluateDisk( tNormalW, dViewDirW, vPositionW, mInv, dLTCCoords );
}

float getDiskLightSpecular() {
    return calcDiskLightSpecular(dNormalW, dLTCUV);
}

#ifdef LIT_CLEARCOAT
float getDiskLightSpecularCC() {
    return calcDiskLightSpecular(ccNormalW, ccLTCUV);
}
#endif

float getSphereLightSpecular() {
    return calcDiskLightSpecular(dNormalW, dLTCUV);
}

#ifdef LIT_CLEARCOAT
float getSphereLightSpecularCC() {
    return calcDiskLightSpecular(ccNormalW, ccLTCUV);
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\metalnessModulate.js
```glsl

uniform float material_f0;

void getMetalnessModulate() {
    vec3 dielectricF0 = material_f0 * dSpecularity;
    dSpecularity = mix(dielectricF0, dAlbedo, dMetalness);
    dAlbedo *= 1.0 - dMetalness;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\outputAlpha.js
```glsl
gl_FragColor.a = dAlpha;
```
engine\src\scene\shader-lib\chunks\lit\frag\outputAlphaOpaque.js
```glsl
    gl_FragColor.a = 1.0;
```
engine\src\scene\shader-lib\chunks\lit\frag\outputAlphaPremul.js
```glsl
gl_FragColor.rgb *= dAlpha;
gl_FragColor.a = dAlpha;
```
engine\src\scene\shader-lib\chunks\lit\frag\reflDir.js
```glsl
void getReflDir() {
    dReflDirW = normalize(-reflect(dViewDirW, dNormalW));
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflDirAniso.js
```glsl
void getReflDir() {
    float roughness = sqrt(1.0 - min(dGlossiness, 1.0));
    float anisotropy = material_anisotropy * roughness;
    vec3 anisotropicDirection = anisotropy >= 0.0 ? dTBN[1] : dTBN[0];
    vec3 anisotropicTangent = cross(anisotropicDirection, dViewDirW);
    vec3 anisotropicNormal = cross(anisotropicTangent, anisotropicDirection);
    vec3 bentNormal = normalize(mix(normalize(dNormalW), normalize(anisotropicNormal), anisotropy));
    dReflDirW = reflect(-dViewDirW, bentNormal);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionCC.js
```glsl
#ifdef LIT_CLEARCOAT
void addReflectionCC() {
    ccReflection += calcReflection(ccReflDirW, ccGlossiness);
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionCube.js
```glsl
uniform samplerCube texture_cubeMap;
uniform float material_reflectivity;

vec3 calcReflection(vec3 tReflDirW, float tGlossiness) {
    vec3 lookupVec = fixSeams(cubeMapProject(tReflDirW));
    lookupVec.x *= -1.0;
    return $DECODE(textureCube(texture_cubeMap, lookupVec));
}

void addReflection() {   
    dReflection += vec4(calcReflection(dReflDirW, dGlossiness), material_reflectivity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionEnv.js
```glsl
#ifndef ENV_ATLAS
#define ENV_ATLAS
uniform sampler2D texture_envAtlas;
#endif
uniform float material_reflectivity;

// calculate mip level for shiny reflection given equirect coords uv.
float shinyMipLevel(vec2 uv) {
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);

    // calculate second dF at 180 degrees
    vec2 uv2 = vec2(fract(uv.x + 0.5), uv.y);
    vec2 dx2 = dFdx(uv2);
    vec2 dy2 = dFdy(uv2);

    // calculate min of both sets of dF to handle discontinuity at the azim edge
    float maxd = min(max(dot(dx, dx), dot(dy, dy)), max(dot(dx2, dx2), dot(dy2, dy2)));

    return clamp(0.5 * log2(maxd) - 1.0 + textureBias, 0.0, 5.0);
}

vec3 calcReflection(vec3 tReflDirW, float tGlossiness) {
    vec3 dir = cubeMapProject(tReflDirW) * vec3(-1.0, 1.0, 1.0);
    vec2 uv = toSphericalUv(dir);

    // calculate roughness level
    float level = saturate(1.0 - tGlossiness) * 5.0;
    float ilevel = floor(level);

    // accessing the shiny (top level) reflection - perform manual mipmap lookup
    float level2 = shinyMipLevel(uv * atlasSize);
    float ilevel2 = floor(level2);

    vec2 uv0, uv1;
    float weight;
    if (ilevel == 0.0) {
        uv0 = mapShinyUv(uv, ilevel2);
        uv1 = mapShinyUv(uv, ilevel2 + 1.0);
        weight = level2 - ilevel2;
    } else {
        // accessing rough reflection - just sample the same part twice
        uv0 = uv1 = mapRoughnessUv(uv, ilevel);
        weight = 0.0;
    }

    vec3 linearA = $DECODE(texture2D(texture_envAtlas, uv0));
    vec3 linearB = $DECODE(texture2D(texture_envAtlas, uv1));
    vec3 linear0 = mix(linearA, linearB, weight);
    vec3 linear1 = $DECODE(texture2D(texture_envAtlas, mapRoughnessUv(uv, ilevel + 1.0)));

    return processEnvironment(mix(linear0, linear1, level - ilevel));
}

void addReflection() {   
    dReflection += vec4(calcReflection(dReflDirW, dGlossiness), material_reflectivity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionEnvHQ.js
```glsl
#ifndef ENV_ATLAS
#define ENV_ATLAS
uniform sampler2D texture_envAtlas;
#endif
uniform samplerCube texture_cubeMap;
uniform float material_reflectivity;

vec3 calcReflection(vec3 tReflDirW, float tGlossiness) {
    vec3 dir = cubeMapProject(tReflDirW) * vec3(-1.0, 1.0, 1.0);
    vec2 uv = toSphericalUv(dir);

    // calculate roughness level
    float level = saturate(1.0 - tGlossiness) * 5.0;
    float ilevel = floor(level);
    float flevel = level - ilevel;

    vec3 sharp = $DECODE(textureCube(texture_cubeMap, fixSeams(dir)));
    vec3 roughA = $DECODE(texture2D(texture_envAtlas, mapRoughnessUv(uv, ilevel)));
    vec3 roughB = $DECODE(texture2D(texture_envAtlas, mapRoughnessUv(uv, ilevel + 1.0)));

    return processEnvironment(mix(sharp, mix(roughA, roughB, flevel), min(level, 1.0)));
}

void addReflection() {   
    dReflection += vec4(calcReflection(dReflDirW, dGlossiness), material_reflectivity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionSheen.js
```glsl

void addReflectionSheen() {
    float NoV = dot(dNormalW, dViewDirW);
    float alphaG = sGlossiness * sGlossiness;

    // Avoid using a LUT and approximate the values analytically
    float a = sGlossiness < 0.25 ? -339.2 * alphaG + 161.4 * sGlossiness - 25.9 : -8.48 * alphaG + 14.3 * sGlossiness - 9.95;
    float b = sGlossiness < 0.25 ? 44.0 * alphaG - 23.7 * sGlossiness + 3.26 : 1.97 * alphaG - 3.27 * sGlossiness + 0.72;
    float DG = exp( a * NoV + b ) + ( sGlossiness < 0.25 ? 0.0 : 0.1 * ( sGlossiness - 0.25 ) );
    sReflection += calcReflection(dNormalW, 0.0) * saturate(DG);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionSphere.js
```glsl
#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif
uniform sampler2D texture_sphereMap;
uniform float material_reflectivity;

vec3 calcReflection(vec3 tReflDirW, float tGlossiness) {
    vec3 reflDirV = (mat3(matrix_view) * tReflDirW).xyz;

    float m = 2.0 * sqrt( dot(reflDirV.xy, reflDirV.xy) + (reflDirV.z+1.0)*(reflDirV.z+1.0) );
    vec2 sphereMapUv = reflDirV.xy / m + 0.5;

    return $DECODE(texture2D(texture_sphereMap, sphereMapUv));
}

void addReflection() {   
    dReflection += vec4(calcReflection(dReflDirW, dGlossiness), material_reflectivity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\reflectionSphereLow.js
```glsl
uniform sampler2D texture_sphereMap;
uniform float material_reflectivity;

vec3 calcReflection(vec3 tReflDirW, float tGlossiness) {
    vec3 reflDirV = vNormalV;

    vec2 sphereMapUv = reflDirV.xy * 0.5 + 0.5;
    return $DECODE(texture2D(texture_sphereMap, sphereMapUv));
}

void addReflection() {   
    dReflection += vec4(calcReflection(dReflDirW, dGlossiness), material_reflectivity);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\refractionCube.js
```glsl
uniform float material_refractionIndex;

vec3 refract2(vec3 viewVec, vec3 Normal, float IOR) {
    float vn = dot(viewVec, Normal);
    float k = 1.0 - IOR * IOR * (1.0 - vn * vn);
    vec3 refrVec = IOR * viewVec - (IOR * vn + sqrt(k)) * Normal;
    return refrVec;
}

void addRefraction() {
    // use same reflection code with refraction vector
    vec3 tmpDir = dReflDirW;
    vec4 tmpRefl = dReflection;
    dReflDirW = refract2(-dViewDirW, dNormalW, material_refractionIndex);
    dReflection = vec4(0);
    addReflection();
    dDiffuseLight = mix(dDiffuseLight, dReflection.rgb * dAlbedo, dTransmission);
    dReflection = tmpRefl;
    dReflDirW = tmpDir;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\refractionDynamic.js
```glsl
uniform float material_refractionIndex;
uniform float material_invAttenuationDistance;
uniform vec3 material_attenuation;

vec3 refract2(vec3 viewVec, vec3 Normal, float IOR) {
    float vn = dot(viewVec, Normal);
    float k = 1.0 - IOR * IOR * (1.0 - vn * vn);
    vec3 refrVec = IOR * viewVec - (IOR * vn + sqrt(k)) * Normal;
    return refrVec;
}

void addRefraction() {

    // Extract scale from the model transform
    vec3 modelScale;
    modelScale.x = length(vec3(matrix_model[0].xyz));
    modelScale.y = length(vec3(matrix_model[1].xyz));
    modelScale.z = length(vec3(matrix_model[2].xyz));

    // Calculate the refraction vector, scaled by the thickness and scale of the object
    vec3 refractionVector = normalize(refract(-dViewDirW, dNormalW, material_refractionIndex)) * dThickness * modelScale;

    // The refraction point is the entry point + vector to exit point
    vec4 pointOfRefraction = vec4(vPositionW + refractionVector, 1.0);

    // Project to texture space so we can sample it
    vec4 projectionPoint = matrix_viewProjection * pointOfRefraction;

    // use built-in getGrabScreenPos function to convert screen position to grab texture uv coords
    vec2 uv = getGrabScreenPos(projectionPoint);

    #ifdef SUPPORTS_TEXLOD
        // Use IOR and roughness to select mip
        float iorToRoughness = (1.0 - dGlossiness) * clamp((1.0 / material_refractionIndex) * 2.0 - 2.0, 0.0, 1.0);
        float refractionLod = log2(uScreenSize.x) * iorToRoughness;
        vec3 refraction = texture2DLodEXT(uSceneColorMap, uv, refractionLod).rgb;
    #else
        vec3 refraction = texture2D(uSceneColorMap, uv).rgb;
    #endif

    // Transmittance is our final refraction color
    vec3 transmittance;
    if (material_invAttenuationDistance != 0.0)
    {
        vec3 attenuation = -log(material_attenuation) * material_invAttenuationDistance;
        transmittance = exp(-attenuation * length(refractionVector));
    }
    else
    {
        transmittance = refraction;
    }

    // Apply fresnel effect on refraction
    vec3 fresnel = vec3(1.0) - getFresnel(dot(dViewDirW, dNormalW), dSpecularity);
    dDiffuseLight = mix(dDiffuseLight, refraction * transmittance * fresnel, dTransmission);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowCascades.js
```glsl
const float maxCascades = 4.0;

// shadow matrix for selected cascade
mat4 cascadeShadowMat;

// function which selects a shadow projection matrix based on cascade distances 
void getShadowCascadeMatrix(mat4 shadowMatrixPalette[4], float shadowCascadeDistances[4], float shadowCascadeCount) {

    // depth in 0 .. far plane range
    float depth = 1.0 / gl_FragCoord.w;

    // find cascade index based on the depth (loop as there is no per component vec compare operator in webgl)
    float cascadeIndex = 0.0;
    for (float i = 0.0; i < maxCascades; i++) {
        if (depth < shadowCascadeDistances[int(i)]) {
            cascadeIndex = i;
            break;
        }
    }

    // limit to actual number of used cascades
    cascadeIndex = min(cascadeIndex, shadowCascadeCount - 1.0);

    // pick shadow matrix
    #ifdef GL2
        cascadeShadowMat = shadowMatrixPalette[int(cascadeIndex)];
    #else
        // webgl 1 does not allow non-cost index array lookup
        if (cascadeIndex == 0.0) {
            cascadeShadowMat = shadowMatrixPalette[0];
        }
        else if (cascadeIndex == 1.0) {
            cascadeShadowMat = shadowMatrixPalette[1];
        }
        else if (cascadeIndex == 2.0) {
            cascadeShadowMat = shadowMatrixPalette[2];
        }
        else {
            cascadeShadowMat = shadowMatrixPalette[3];
        }
    #endif
}

void fadeShadow(float shadowCascadeDistances[4]) {                  

    // if the pixel is past the shadow distance, remove shadow
    // this enforces straight line instead of corner of shadow which moves when camera rotates  
    float depth = 1.0 / gl_FragCoord.w;
    if (depth > shadowCascadeDistances[int(maxCascades - 1.0)]) {
        dShadowCoord.z = -9999999.0;
    }
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowCommon.js
```glsl
void normalOffsetPointShadow(vec4 shadowParams) {
    float distScale = length(dLightDirW);
    vec3 wPos = vPositionW + dVertexNormalW * shadowParams.y * clamp(1.0 - dot(dVertexNormalW, -dLightDirNormW), 0.0, 1.0) * distScale; //0.02
    vec3 dir = wPos - dLightPosW;
    dLightDirW = dir;
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowCoord.js
```glsl
void _getShadowCoordOrtho(mat4 shadowMatrix, vec3 shadowParams, vec3 wPos) {
    dShadowCoord = (shadowMatrix * vec4(wPos, 1.0)).xyz;
    dShadowCoord.z = saturate(dShadowCoord.z) - 0.0001;

    #ifdef SHADOWBIAS
    dShadowCoord.z += getShadowBias(shadowParams.x, shadowParams.z);
    #endif
}

void _getShadowCoordPersp(mat4 shadowMatrix, vec4 shadowParams, vec3 wPos) {
    vec4 projPos = shadowMatrix * vec4(wPos, 1.0);
    projPos.xy /= projPos.w;
    dShadowCoord.xy = projPos.xy;
    dShadowCoord.z = length(dLightDirW) * shadowParams.w;

    #ifdef SHADOWBIAS
    dShadowCoord.z += getShadowBias(shadowParams.x, shadowParams.z);
    #endif
}

void getShadowCoordOrtho(mat4 shadowMatrix, vec3 shadowParams) {
    _getShadowCoordOrtho(shadowMatrix, shadowParams, vPositionW);
}

void getShadowCoordPersp(mat4 shadowMatrix, vec4 shadowParams) {
    _getShadowCoordPersp(shadowMatrix, shadowParams, vPositionW);
}

void getShadowCoordPerspNormalOffset(mat4 shadowMatrix, vec4 shadowParams) {
    float distScale = abs(dot(vPositionW - dLightPosW, dLightDirNormW)); // fov?
    vec3 wPos = vPositionW + dVertexNormalW * shadowParams.y * clamp(1.0 - dot(dVertexNormalW, -dLightDirNormW), 0.0, 1.0) * distScale;

    _getShadowCoordPersp(shadowMatrix, shadowParams, wPos);
}

void getShadowCoordOrthoNormalOffset(mat4 shadowMatrix, vec3 shadowParams) {
    vec3 wPos = vPositionW + dVertexNormalW * shadowParams.y * clamp(1.0 - dot(dVertexNormalW, -dLightDirNormW), 0.0, 1.0); //0.08

    _getShadowCoordOrtho(shadowMatrix, shadowParams, wPos);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowCoordPerspZbuffer.js
```glsl
void _getShadowCoordPerspZbuffer(mat4 shadowMatrix, vec4 shadowParams, vec3 wPos) {
    vec4 projPos = shadowMatrix * vec4(wPos, 1.0);
    projPos.xyz /= projPos.w;
    dShadowCoord = projPos.xyz;
    // depth bias is already applied on render
}

void getShadowCoordPerspZbufferNormalOffset(mat4 shadowMatrix, vec4 shadowParams) {
    vec3 wPos = vPositionW + dVertexNormalW * shadowParams.y;
    _getShadowCoordPerspZbuffer(shadowMatrix, shadowParams, wPos);
}

void getShadowCoordPerspZbuffer(mat4 shadowMatrix, vec4 shadowParams) {
    _getShadowCoordPerspZbuffer(shadowMatrix, shadowParams, vPositionW);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowEVSM.js
```glsl
float VSM$(sampler2D tex, vec2 texCoords, float resolution, float Z, float vsmBias, float exponent) {
    vec3 moments = texture2D(tex, texCoords).xyz;
    return calculateEVSM(moments, Z, vsmBias, exponent);
}

float getShadowVSM$(sampler2D shadowMap, vec3 shadowParams, float exponent) {
    return VSM$(shadowMap, dShadowCoord.xy, shadowParams.x, dShadowCoord.z, shadowParams.y, exponent);
}

float getShadowSpotVSM$(sampler2D shadowMap, vec4 shadowParams, float exponent) {
    return VSM$(shadowMap, dShadowCoord.xy, shadowParams.x, length(dLightDirW) * shadowParams.w + shadowParams.z, shadowParams.y, exponent);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowEVSMn.js
```glsl
float VSM$(sampler2D tex, vec2 texCoords, float resolution, float Z, float vsmBias, float exponent) {
    float pixelSize = 1.0 / resolution;
    texCoords -= vec2(pixelSize);
    vec3 s00 = texture2D(tex, texCoords).xyz;
    vec3 s10 = texture2D(tex, texCoords + vec2(pixelSize, 0)).xyz;
    vec3 s01 = texture2D(tex, texCoords + vec2(0, pixelSize)).xyz;
    vec3 s11 = texture2D(tex, texCoords + vec2(pixelSize)).xyz;
    vec2 fr = fract(texCoords * resolution);
    vec3 h0 = mix(s00, s10, fr.x);
    vec3 h1 = mix(s01, s11, fr.x);
    vec3 moments = mix(h0, h1, fr.y);
    return calculateEVSM(moments, Z, vsmBias, exponent);
}

float getShadowVSM$(sampler2D shadowMap, vec3 shadowParams, float exponent) {
    return VSM$(shadowMap, dShadowCoord.xy, shadowParams.x, dShadowCoord.z, shadowParams.y, exponent);
}

float getShadowSpotVSM$(sampler2D shadowMap, vec4 shadowParams, float exponent) {
    return VSM$(shadowMap, dShadowCoord.xy, shadowParams.x, length(dLightDirW) * shadowParams.w + shadowParams.z, shadowParams.y, exponent);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowStandard.js
```glsl
vec3 lessThan2(vec3 a, vec3 b) {
    return clamp((b - a)*1000.0, 0.0, 1.0); // softer version
}

#ifndef UNPACKFLOAT
#define UNPACKFLOAT
float unpackFloat(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0 / (256.0 * 256.0 * 256.0), 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0);
    return dot(rgbaDepth, bitShift);
}
#endif

// ----- Direct/Spot Sampling -----

#ifdef GL2

float _getShadowPCF3x3(SHADOWMAP_ACCEPT(shadowMap), vec3 shadowParams) {
    float z = dShadowCoord.z;
    vec2 uv = dShadowCoord.xy * shadowParams.x; // 1 unit - 1 texel
    float shadowMapSizeInv = 1.0 / shadowParams.x;
    vec2 base_uv = floor(uv + 0.5);
    float s = (uv.x + 0.5 - base_uv.x);
    float t = (uv.y + 0.5 - base_uv.y);
    base_uv -= vec2(0.5);
    base_uv *= shadowMapSizeInv;

    float sum = 0.0;

    float uw0 = (3.0 - 2.0 * s);
    float uw1 = (1.0 + 2.0 * s);

    float u0 = (2.0 - s) / uw0 - 1.0;
    float u1 = s / uw1 + 1.0;

    float vw0 = (3.0 - 2.0 * t);
    float vw1 = (1.0 + 2.0 * t);

    float v0 = (2.0 - t) / vw0 - 1.0;
    float v1 = t / vw1 + 1.0;

    u0 = u0 * shadowMapSizeInv + base_uv.x;
    v0 = v0 * shadowMapSizeInv + base_uv.y;

    u1 = u1 * shadowMapSizeInv + base_uv.x;
    v1 = v1 * shadowMapSizeInv + base_uv.y;

    sum += uw0 * vw0 * textureShadow(shadowMap, vec3(u0, v0, z));
    sum += uw1 * vw0 * textureShadow(shadowMap, vec3(u1, v0, z));
    sum += uw0 * vw1 * textureShadow(shadowMap, vec3(u0, v1, z));
    sum += uw1 * vw1 * textureShadow(shadowMap, vec3(u1, v1, z));

    sum *= 1.0f / 16.0;
    return sum;
}

float getShadowPCF3x3(SHADOWMAP_ACCEPT(shadowMap), vec3 shadowParams) {
    return _getShadowPCF3x3(SHADOWMAP_PASS(shadowMap), shadowParams);
}

float getShadowSpotPCF3x3(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams) {
    return _getShadowPCF3x3(SHADOWMAP_PASS(shadowMap), shadowParams.xyz);
}

#else // GL1

float _xgetShadowPCF3x3(mat3 depthKernel, sampler2D shadowMap, vec3 shadowParams) {
    mat3 shadowKernel;
    vec3 shadowCoord = dShadowCoord;
    vec3 shadowZ = vec3(shadowCoord.z);
    shadowKernel[0] = vec3(greaterThan(depthKernel[0], shadowZ));
    shadowKernel[1] = vec3(greaterThan(depthKernel[1], shadowZ));
    shadowKernel[2] = vec3(greaterThan(depthKernel[2], shadowZ));

    vec2 fractionalCoord = fract( shadowCoord.xy * shadowParams.x );

    shadowKernel[0] = mix(shadowKernel[0], shadowKernel[1], fractionalCoord.x);
    shadowKernel[1] = mix(shadowKernel[1], shadowKernel[2], fractionalCoord.x);

    vec4 shadowValues;
    shadowValues.x = mix(shadowKernel[0][0], shadowKernel[0][1], fractionalCoord.y);
    shadowValues.y = mix(shadowKernel[0][1], shadowKernel[0][2], fractionalCoord.y);
    shadowValues.z = mix(shadowKernel[1][0], shadowKernel[1][1], fractionalCoord.y);
    shadowValues.w = mix(shadowKernel[1][1], shadowKernel[1][2], fractionalCoord.y);

    return dot( shadowValues, vec4( 1.0 ) ) * 0.25;
}

float _getShadowPCF3x3(sampler2D shadowMap, vec3 shadowParams) {
    vec3 shadowCoord = dShadowCoord;

    float xoffset = 1.0 / shadowParams.x; // 1/shadow map width
    float dx0 = -xoffset;
    float dx1 = xoffset;

    mat3 depthKernel;
    depthKernel[0][0] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx0, dx0)));
    depthKernel[0][1] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx0, 0.0)));
    depthKernel[0][2] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx0, dx1)));
    depthKernel[1][0] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(0.0, dx0)));
    depthKernel[1][1] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy));
    depthKernel[1][2] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(0.0, dx1)));
    depthKernel[2][0] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx1, dx0)));
    depthKernel[2][1] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx1, 0.0)));
    depthKernel[2][2] = unpackFloat(textureShadow(shadowMap, shadowCoord.xy + vec2(dx1, dx1)));

    return _xgetShadowPCF3x3(depthKernel, shadowMap, shadowParams);
}

float getShadowPCF3x3(sampler2D shadowMap, vec3 shadowParams) {
    return _getShadowPCF3x3(shadowMap, shadowParams);
}

float getShadowSpotPCF3x3(sampler2D shadowMap, vec4 shadowParams) {
    return _getShadowPCF3x3(shadowMap, shadowParams.xyz);
}
#endif


// ----- Omni Sampling -----

#ifndef WEBGPU

float _getShadowPoint(samplerCube shadowMap, vec4 shadowParams, vec3 dir) {

    vec3 tc = normalize(dir);
    vec3 tcAbs = abs(tc);

    vec4 dirX = vec4(1,0,0, tc.x);
    vec4 dirY = vec4(0,1,0, tc.y);
    float majorAxisLength = tc.z;
    if ((tcAbs.x > tcAbs.y) && (tcAbs.x > tcAbs.z)) {
        dirX = vec4(0,0,1, tc.z);
        dirY = vec4(0,1,0, tc.y);
        majorAxisLength = tc.x;
    } else if ((tcAbs.y > tcAbs.x) && (tcAbs.y > tcAbs.z)) {
        dirX = vec4(1,0,0, tc.x);
        dirY = vec4(0,0,1, tc.z);
        majorAxisLength = tc.y;
    }

    float shadowParamsInFaceSpace = ((1.0/shadowParams.x) * 2.0) * abs(majorAxisLength);

    vec3 xoffset = (dirX.xyz * shadowParamsInFaceSpace);
    vec3 yoffset = (dirY.xyz * shadowParamsInFaceSpace);
    vec3 dx0 = -xoffset;
    vec3 dy0 = -yoffset;
    vec3 dx1 = xoffset;
    vec3 dy1 = yoffset;

    mat3 shadowKernel;
    mat3 depthKernel;

    depthKernel[0][0] = unpackFloat(textureCube(shadowMap, tc + dx0 + dy0));
    depthKernel[0][1] = unpackFloat(textureCube(shadowMap, tc + dx0));
    depthKernel[0][2] = unpackFloat(textureCube(shadowMap, tc + dx0 + dy1));
    depthKernel[1][0] = unpackFloat(textureCube(shadowMap, tc + dy0));
    depthKernel[1][1] = unpackFloat(textureCube(shadowMap, tc));
    depthKernel[1][2] = unpackFloat(textureCube(shadowMap, tc + dy1));
    depthKernel[2][0] = unpackFloat(textureCube(shadowMap, tc + dx1 + dy0));
    depthKernel[2][1] = unpackFloat(textureCube(shadowMap, tc + dx1));
    depthKernel[2][2] = unpackFloat(textureCube(shadowMap, tc + dx1 + dy1));

    vec3 shadowZ = vec3(length(dir) * shadowParams.w + shadowParams.z);

    shadowKernel[0] = vec3(lessThan2(depthKernel[0], shadowZ));
    shadowKernel[1] = vec3(lessThan2(depthKernel[1], shadowZ));
    shadowKernel[2] = vec3(lessThan2(depthKernel[2], shadowZ));

    vec2 uv = (vec2(dirX.w, dirY.w) / abs(majorAxisLength)) * 0.5;

    vec2 fractionalCoord = fract( uv * shadowParams.x );

    shadowKernel[0] = mix(shadowKernel[0], shadowKernel[1], fractionalCoord.x);
    shadowKernel[1] = mix(shadowKernel[1], shadowKernel[2], fractionalCoord.x);

    vec4 shadowValues;
    shadowValues.x = mix(shadowKernel[0][0], shadowKernel[0][1], fractionalCoord.y);
    shadowValues.y = mix(shadowKernel[0][1], shadowKernel[0][2], fractionalCoord.y);
    shadowValues.z = mix(shadowKernel[1][0], shadowKernel[1][1], fractionalCoord.y);
    shadowValues.w = mix(shadowKernel[1][1], shadowKernel[1][2], fractionalCoord.y);

    return 1.0 - dot( shadowValues, vec4( 1.0 ) ) * 0.25;
}

float getShadowPointPCF3x3(samplerCube shadowMap, vec4 shadowParams) {
    return _getShadowPoint(shadowMap, shadowParams, dLightDirW);
}

#endif
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowStandardGL2.js
```glsl
float _getShadowPCF5x5(SHADOWMAP_ACCEPT(shadowMap), vec3 shadowParams) {
    // http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/

    float z = dShadowCoord.z;
    vec2 uv = dShadowCoord.xy * shadowParams.x; // 1 unit - 1 texel
    float shadowMapSizeInv = 1.0 / shadowParams.x;
    vec2 base_uv = floor(uv + 0.5);
    float s = (uv.x + 0.5 - base_uv.x);
    float t = (uv.y + 0.5 - base_uv.y);
    base_uv -= vec2(0.5);
    base_uv *= shadowMapSizeInv;


    float uw0 = (4.0 - 3.0 * s);
    float uw1 = 7.0;
    float uw2 = (1.0 + 3.0 * s);

    float u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    float u1 = (3.0 + s) / uw1;
    float u2 = s / uw2 + 2.0;

    float vw0 = (4.0 - 3.0 * t);
    float vw1 = 7.0;
    float vw2 = (1.0 + 3.0 * t);

    float v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    float v1 = (3.0 + t) / vw1;
    float v2 = t / vw2 + 2.0;

    float sum = 0.0;

    u0 = u0 * shadowMapSizeInv + base_uv.x;
    v0 = v0 * shadowMapSizeInv + base_uv.y;

    u1 = u1 * shadowMapSizeInv + base_uv.x;
    v1 = v1 * shadowMapSizeInv + base_uv.y;

    u2 = u2 * shadowMapSizeInv + base_uv.x;
    v2 = v2 * shadowMapSizeInv + base_uv.y;

    sum += uw0 * vw0 * textureShadow(shadowMap, vec3(u0, v0, z));
    sum += uw1 * vw0 * textureShadow(shadowMap, vec3(u1, v0, z));
    sum += uw2 * vw0 * textureShadow(shadowMap, vec3(u2, v0, z));

    sum += uw0 * vw1 * textureShadow(shadowMap, vec3(u0, v1, z));
    sum += uw1 * vw1 * textureShadow(shadowMap, vec3(u1, v1, z));
    sum += uw2 * vw1 * textureShadow(shadowMap, vec3(u2, v1, z));

    sum += uw0 * vw2 * textureShadow(shadowMap, vec3(u0, v2, z));
    sum += uw1 * vw2 * textureShadow(shadowMap, vec3(u1, v2, z));
    sum += uw2 * vw2 * textureShadow(shadowMap, vec3(u2, v2, z));

    sum *= 1.0f / 144.0;

    sum = saturate(sum);

    return sum;
}

float getShadowPCF5x5(SHADOWMAP_ACCEPT(shadowMap), vec3 shadowParams) {
    return _getShadowPCF5x5(SHADOWMAP_PASS(shadowMap), shadowParams);
}

float getShadowSpotPCF5x5(SHADOWMAP_ACCEPT(shadowMap), vec4 shadowParams) {
    return _getShadowPCF5x5(SHADOWMAP_PASS(shadowMap), shadowParams.xyz);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowVSM8.js
```glsl
float calculateVSM8(vec3 moments, float Z, float vsmBias) {
    float VSMBias = vsmBias;//0.01 * 0.25;
    float depthScale = VSMBias * Z;
    float minVariance1 = depthScale * depthScale;
    return chebyshevUpperBound(moments.xy, Z, minVariance1, 0.1);
}

float decodeFloatRG(vec2 rg) {
    return rg.y*(1.0/255.0) + rg.x;
}

float VSM8(sampler2D tex, vec2 texCoords, float resolution, float Z, float vsmBias, float exponent) {
    vec4 c = texture2D(tex, texCoords);
    vec3 moments = vec3(decodeFloatRG(c.xy), decodeFloatRG(c.zw), 0.0);
    return calculateVSM8(moments, Z, vsmBias);
}

float getShadowVSM8(sampler2D shadowMap, vec3 shadowParams, float exponent) {
    return VSM8(shadowMap, dShadowCoord.xy, shadowParams.x, dShadowCoord.z, shadowParams.y, 0.0);
}

float getShadowSpotVSM8(sampler2D shadowMap, vec4 shadowParams, float exponent) {
    return VSM8(shadowMap, dShadowCoord.xy, shadowParams.x, length(dLightDirW) * shadowParams.w + shadowParams.z, shadowParams.y, 0.0);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\shadowVSM_common.js
```glsl
float linstep(float a, float b, float v) {
    return saturate((v - a) / (b - a));
}

float reduceLightBleeding(float pMax, float amount) {
   // Remove the [0, amount] tail and linearly rescale (amount, 1].
   return linstep(amount, 1.0, pMax);
}

float chebyshevUpperBound(vec2 moments, float mean, float minVariance, float lightBleedingReduction) {
    // Compute variance
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, minVariance);

    // Compute probabilistic upper bound
    float d = mean - moments.x;
    float pMax = variance / (variance + (d * d));

    pMax = reduceLightBleeding(pMax, lightBleedingReduction);

    // One-tailed Chebyshev
    return (mean <= moments.x ? 1.0 : pMax);
}

float calculateEVSM(vec3 moments, float Z, float vsmBias, float exponent) {
    Z = 2.0 * Z - 1.0;
    float warpedDepth = exp(exponent * Z);

    moments.xy += vec2(warpedDepth, warpedDepth*warpedDepth) * (1.0 - moments.z);

    float VSMBias = vsmBias;//0.01 * 0.25;
    float depthScale = VSMBias * exponent * warpedDepth;
    float minVariance1 = depthScale * depthScale;
    return chebyshevUpperBound(moments.xy, warpedDepth, minVariance1, 0.1);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\spot.js
```glsl
float getSpotEffect(vec3 lightSpotDirW, float lightInnerConeAngle, float lightOuterConeAngle) {
    float cosAngle = dot(dLightDirNormW, lightSpotDirW);
    return smoothstep(lightOuterConeAngle, lightInnerConeAngle, cosAngle);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\start.js
```glsl
void main(void) {
    dReflection = vec4(0);

    #ifdef LIT_CLEARCOAT
    ccSpecularLight = vec3(0);
    ccReflection = vec3(0);
    #endif
```
engine\src\scene\shader-lib\chunks\lit\frag\startNineSliced.js
```glsl
    nineSlicedUv = vUv0;
    nineSlicedUv.y = 1.0 - nineSlicedUv.y;

```
engine\src\scene\shader-lib\chunks\lit\frag\startNineSlicedTiled.js
```glsl
    vec2 tileMask = step(vMask, vec2(0.99999));
    vec2 tileSize = 0.5 * (innerOffset.xy + innerOffset.zw);
    vec2 tileScale = vec2(1.0) / (vec2(1.0) - tileSize);
    vec2 clampedUv = mix(innerOffset.xy * 0.5, vec2(1.0) - innerOffset.zw * 0.5, fract((vTiledUv - tileSize) * tileScale));
    clampedUv = clampedUv * atlasRect.zw + atlasRect.xy;
    nineSlicedUv = vUv0 * tileMask + clampedUv * (vec2(1.0) - tileMask);
    nineSlicedUv.y = 1.0 - nineSlicedUv.y;
    
```
engine\src\scene\shader-lib\chunks\lit\frag\storeEVSM.js
```glsl
float exponent = VSM_EXPONENT;

depth = 2.0 * depth - 1.0;
depth =  exp(exponent * depth);
gl_FragColor = vec4(depth, depth*depth, 1.0, 1.0);
```
engine\src\scene\shader-lib\chunks\lit\frag\TBN.js
```glsl
void getTBN() {
    dTBN = mat3(normalize(dTangentW), normalize(dBinormalW), normalize(dVertexNormalW));
}
```
engine\src\scene\shader-lib\chunks\lit\frag\TBNderivative.js
```glsl
uniform float tbnBasis;

// http://www.thetenthplanet.de/archives/1180
void getTBN() {
    vec2 uv = $UV;

    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( vPositionW );
    vec3 dp2 = dFdy( vPositionW );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // solve the linear system
    vec3 dp2perp = cross( dp2, dVertexNormalW );
    vec3 dp1perp = cross( dVertexNormalW, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float denom = max( dot(T,T), dot(B,B) );
    float invmax = (denom == 0.0) ? 0.0 : tbnBasis / sqrt( denom );
    dTBN = mat3(T * invmax, -B * invmax, dVertexNormalW );
}
```
engine\src\scene\shader-lib\chunks\lit\frag\TBNfast.js
```glsl
void getTBN() {
    dTBN = mat3(dTangentW, dBinormalW, dVertexNormalW);
}
```
engine\src\scene\shader-lib\chunks\lit\frag\TBNObjectSpace.js
```glsl
void getTBN() {

    vec3 B = cross(dVertexNormalW, vObjectSpaceUpW);
    vec3 T = cross(dVertexNormalW, B);

    if (dot(B,B)==0.0) // deal with case when vObjectSpaceUpW dVertexNormalW are parallel
    {
        float major=max(max(dVertexNormalW.x, dVertexNormalW.y),dVertexNormalW.z);

        if (dVertexNormalW.x==major)
        {
            B=cross(dVertexNormalW, vec3(0,1,0));
            T=cross(dVertexNormalW, B);
        }
        else if (dVertexNormalW.y==major)
        {
            B=cross(dVertexNormalW, vec3(0,0,1));
            T=cross(dVertexNormalW, B);
        }
        else if (dVertexNormalW.z==major)
        {
            B=cross(dVertexNormalW, vec3(1,0,0));
            T=cross(dVertexNormalW, B);
        }
    }

    dTBN = mat3(normalize(T), normalize(B), normalize(dVertexNormalW));
}
```
engine\src\scene\shader-lib\chunks\lit\frag\viewDir.js
```glsl
void getViewDir() {
    dViewDirW = normalize(view_position - vPositionW);
}
```
engine\src\scene\shader-lib\chunks\lit\vert\base.js
```glsl
attribute vec3 vertex_position;
attribute vec3 vertex_normal;
attribute vec4 vertex_tangent;
attribute vec2 vertex_texCoord0;
attribute vec2 vertex_texCoord1;
attribute vec4 vertex_color;

uniform mat4 matrix_viewProjection;
uniform mat4 matrix_model;
uniform mat3 matrix_normal;

vec3 dPositionW;
mat4 dModelMatrix;
mat3 dNormalMatrix;
```
engine\src\scene\shader-lib\chunks\lit\vert\baseNineSliced.js
```glsl
#define NINESLICED

varying vec2 vMask;
varying vec2 vTiledUv;

uniform mediump vec4 innerOffset;
uniform mediump vec2 outerScale;
uniform mediump vec4 atlasRect;
```
engine\src\scene\shader-lib\chunks\lit\vert\instancing.js
```glsl
attribute vec4 instance_line1;
attribute vec4 instance_line2;
attribute vec4 instance_line3;
attribute vec4 instance_line4;
```
engine\src\scene\shader-lib\chunks\lit\vert\normal.js
```glsl
#ifdef MORPHING_TEXTURE_BASED_NORMAL
uniform highp sampler2D morphNormalTex;
#endif

vec3 getNormal() {
    #ifdef SKIN
    dNormalMatrix = mat3(dModelMatrix[0].xyz, dModelMatrix[1].xyz, dModelMatrix[2].xyz);
    #elif defined(INSTANCING)
    dNormalMatrix = mat3(instance_line1.xyz, instance_line2.xyz, instance_line3.xyz);
    #else
    dNormalMatrix = matrix_normal;
    #endif

    vec3 tempNormal = vertex_normal;

    #ifdef MORPHING
    #ifdef MORPHING_NRM03
    tempNormal += morph_weights_a[0] * morph_nrm0;
    tempNormal += morph_weights_a[1] * morph_nrm1;
    tempNormal += morph_weights_a[2] * morph_nrm2;
    tempNormal += morph_weights_a[3] * morph_nrm3;
    #endif
    #ifdef MORPHING_NRM47
    tempNormal += morph_weights_b[0] * morph_nrm4;
    tempNormal += morph_weights_b[1] * morph_nrm5;
    tempNormal += morph_weights_b[2] * morph_nrm6;
    tempNormal += morph_weights_b[3] * morph_nrm7;
    #endif
    #endif

    #ifdef MORPHING_TEXTURE_BASED_NORMAL
    // apply morph offset from texture
    vec2 morphUV = getTextureMorphCoords();
    vec3 morphNormal = texture2D(morphNormalTex, morphUV).xyz;
    tempNormal += morphNormal;
    #endif

    return normalize(dNormalMatrix * tempNormal);
}
```
engine\src\scene\shader-lib\chunks\lit\vert\normalInstanced.js
```glsl
vec3 getNormal() {
    dNormalMatrix = mat3(instance_line1.xyz, instance_line2.xyz, instance_line3.xyz);
    return normalize(dNormalMatrix * vertex_normal);
}
```
engine\src\scene\shader-lib\chunks\lit\vert\normalSkinned.js
```glsl
vec3 getNormal() {
    dNormalMatrix = mat3(dModelMatrix[0].xyz, dModelMatrix[1].xyz, dModelMatrix[2].xyz);
    return normalize(dNormalMatrix * vertex_normal);
}
```
engine\src\scene\shader-lib\chunks\lit\vert\start.js
```glsl
void main(void) {
    gl_Position = getPosition();
```
engine\src\scene\shader-lib\chunks\lit\vert\tangentBinormal.js
```glsl
vec3 getTangent() {
    return normalize(dNormalMatrix * vertex_tangent.xyz);
}

vec3 getBinormal() {
    return cross(vNormalW, vTangentW) * vertex_tangent.w;
}

vec3 getObjectSpaceUp() {
    return normalize(dNormalMatrix * vec3(0, 1, 0));
}
```
engine\src\scene\shader-lib\chunks\lit\vert\uv0.js
```glsl
#ifdef NINESLICED
vec2 getUv0() {
    vec2 uv = vertex_position.xz;

    // offset inner vertices inside
    // (original vertices must be in [-1;1] range)
    vec2 positiveUnitOffset = clamp(vertex_position.xz, vec2(0.0), vec2(1.0));
    vec2 negativeUnitOffset = clamp(-vertex_position.xz, vec2(0.0), vec2(1.0));
    uv += (-positiveUnitOffset * innerOffset.xy + negativeUnitOffset * innerOffset.zw) * vertex_texCoord0.xy;

    uv = uv * -0.5 + 0.5;
    uv = uv * atlasRect.zw + atlasRect.xy;

    vMask = vertex_texCoord0.xy;

    return uv;
}
#else
vec2 getUv0() {
    return vertex_texCoord0;
}
#endif
```
engine\src\scene\shader-lib\chunks\lit\vert\uv1.js
```glsl
vec2 getUv1() {
    return vertex_texCoord1;
}
```
engine\src\scene\shader-lib\chunks\lit\vert\viewNormal.js
```glsl
#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif

vec3 getViewNormal() {
    return mat3(matrix_view) * vNormalW;
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particle.js
```glsl
varying vec4 texCoordsAlphaLife;

uniform sampler2D colorMap;
uniform sampler2D colorParam;
uniform float graphSampleSize;
uniform float graphNumSamples;

#ifndef CAMERAPLANES
#define CAMERAPLANES
uniform vec4 camera_params;
#endif

uniform float softening;
uniform float colorMult;

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

#ifndef UNPACKFLOAT
#define UNPACKFLOAT
float unpackFloat(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0 / (256.0 * 256.0 * 256.0), 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0);
    float depth = dot(rgbaDepth, bitShift);
    return depth;
}
#endif

void main(void) {
    vec4 tex  = gammaCorrectInput(texture2D(colorMap, vec2(texCoordsAlphaLife.x, 1.0 - texCoordsAlphaLife.y)));
    vec4 ramp = gammaCorrectInput(texture2D(colorParam, vec2(texCoordsAlphaLife.w, 0.0)));
    ramp.rgb *= colorMult;

    ramp.a += texCoordsAlphaLife.z;

    vec3 rgb = tex.rgb * ramp.rgb;
    float a  = tex.a * ramp.a;
```
engine\src\scene\shader-lib\chunks\particle\frag\particleInputFloat.js
```glsl
void readInput(float uv) {
    vec4 tex = texture2D(particleTexIN, vec2(uv, 0.25));
    vec4 tex2 = texture2D(particleTexIN, vec2(uv, 0.75));

    inPos = tex.xyz;
    inVel = tex2.xyz;
    inAngle = (tex.w < 0.0? -tex.w : tex.w) - 1000.0;
    inShow = tex.w >= 0.0;
    inLife = tex2.w;
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleInputRgba8.js
```glsl
//RG=X, BA=Y
//RG=Z, BA=A
//RGB=V, A=visMode
//RGBA=life

#define PI2 6.283185307179586

uniform vec3 inBoundsSize;
uniform vec3 inBoundsCenter;

uniform float maxVel;

float decodeFloatRG(vec2 rg) {
    return rg.y*(1.0/255.0) + rg.x;
}

float decodeFloatRGBA( vec4 rgba ) {
  return dot( rgba, vec4(1.0, 1.0/255.0, 1.0/65025.0, 1.0/160581375.0) );
}

void readInput(float uv) {
    vec4 tex0 = texture2D(particleTexIN, vec2(uv, 0.125));
    vec4 tex1 = texture2D(particleTexIN, vec2(uv, 0.375));
    vec4 tex2 = texture2D(particleTexIN, vec2(uv, 0.625));
    vec4 tex3 = texture2D(particleTexIN, vec2(uv, 0.875));

    inPos = vec3(decodeFloatRG(tex0.rg), decodeFloatRG(tex0.ba), decodeFloatRG(tex1.rg));
    inPos = (inPos - vec3(0.5)) * inBoundsSize + inBoundsCenter;

    inVel = tex2.xyz;
    inVel = (inVel - vec3(0.5)) * maxVel;

    inAngle = decodeFloatRG(tex1.ba) * PI2;
    inShow = tex2.a > 0.5;

    inLife = decodeFloatRGBA(tex3);
    float maxNegLife = max(lifetime, (numParticles - 1.0) * (rate+rateDiv));
    float maxPosLife = lifetime+1.0;
    inLife = inLife * (maxNegLife + maxPosLife) - maxNegLife;
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleOutputFloat.js
```glsl
void writeOutput() {
    if (gl_FragCoord.y<1.0) {
        gl_FragColor = vec4(outPos, (outAngle + 1000.0) * visMode);
    } else {
        gl_FragColor = vec4(outVel, outLife);
    }
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleOutputRgba8.js
```glsl
uniform vec3 outBoundsMul;
uniform vec3 outBoundsAdd;

vec2 encodeFloatRG( float v ) {
    vec2 enc = vec2(1.0, 255.0) * v;
    enc = fract(enc);
    enc -= enc.yy * vec2(1.0/255.0, 1.0/255.0);
    return enc;
}

vec4 encodeFloatRGBA( float v ) {
    vec4 enc = vec4(1.0, 255.0, 65025.0, 160581375.0) * v;
    enc = fract(enc);
    enc -= enc.yzww * vec4(1.0/255.0,1.0/255.0,1.0/255.0,0.0);
    return enc;
}

void writeOutput() {
    outPos = outPos * outBoundsMul + outBoundsAdd;
    outAngle = fract(outAngle / PI2);

    outVel = (outVel / maxVel) + vec3(0.5); // TODO: mul

    float maxNegLife = max(lifetime, (numParticles - 1.0) * (rate+rateDiv));
    float maxPosLife = lifetime+1.0;
    outLife = (outLife + maxNegLife) / (maxNegLife + maxPosLife);

    if (gl_FragCoord.y < 1.0) {
        gl_FragColor = vec4(encodeFloatRG(outPos.x), encodeFloatRG(outPos.y));
    } else if (gl_FragCoord.y < 2.0) {
        gl_FragColor = vec4(encodeFloatRG(outPos.z), encodeFloatRG(outAngle));
    } else if (gl_FragCoord.y < 3.0) {
        gl_FragColor = vec4(outVel, visMode*0.5+0.5);
    } else {
        gl_FragColor = encodeFloatRGBA(outLife);
    }
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterAABB.js
```glsl
uniform mat3 spawnBounds;
uniform vec3 spawnPosInnerRatio;

vec3 calcSpawnPosition(vec3 inBounds, float rndFactor) {
    vec3 pos = inBounds - vec3(0.5);

    vec3 posAbs = abs(pos);
    vec3 maxPos = vec3(max(posAbs.x, max(posAbs.y, posAbs.z)));

    vec3 edge = maxPos + (vec3(0.5) - maxPos) * spawnPosInnerRatio;

    pos.x = edge.x * (maxPos.x == posAbs.x ? sign(pos.x) : 2.0 * pos.x);
    pos.y = edge.y * (maxPos.y == posAbs.y ? sign(pos.y) : 2.0 * pos.y);
    pos.z = edge.z * (maxPos.z == posAbs.z ? sign(pos.z) : 2.0 * pos.z);

#ifndef LOCAL_SPACE
    return emitterPos + spawnBounds * pos;
#else
    return spawnBounds * pos;
#endif
}

void addInitialVelocity(inout vec3 localVelocity, vec3 inBounds) {
    localVelocity -= vec3(0, 0, initialVelocity);
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterEnd.js
```glsl
    writeOutput();
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterInit.js
```glsl
varying vec2 vUv0;

uniform highp sampler2D particleTexIN;
uniform highp sampler2D internalTex0;
uniform highp sampler2D internalTex1;
uniform highp sampler2D internalTex2;
uniform highp sampler2D internalTex3;

uniform mat3 emitterMatrix, emitterMatrixInv;
uniform vec3 emitterScale;

uniform vec3 emitterPos, frameRandom, localVelocityDivMult, velocityDivMult;
uniform float delta, rate, rateDiv, lifetime, numParticles, rotSpeedDivMult, radialSpeedDivMult, seed;
uniform float startAngle, startAngle2;
uniform float initialVelocity;

uniform float graphSampleSize;
uniform float graphNumSamples;

vec3 inPos;
vec3 inVel;
float inAngle;
bool inShow;
float inLife;
float visMode;

vec3 outPos;
vec3 outVel;
float outAngle;
bool outShow;
float outLife;
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterNoRespawn.js
```glsl
    if (outLife >= lifetime) {
        outLife -= max(lifetime, (numParticles - 1.0) * particleRate);
        visMode = -1.0;
    }
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterOnStop.js
```glsl
    visMode = outLife < 0.0? -1.0: visMode;
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterRespawn.js
```glsl
    if (outLife >= lifetime) {
        outLife -= max(lifetime, (numParticles - 1.0) * particleRate);
        visMode = 1.0;
    }
    visMode = outLife < 0.0? 1.0: visMode;
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterSphere.js
```glsl
uniform float spawnBoundsSphere;
uniform float spawnBoundsSphereInnerRatio;

vec3 calcSpawnPosition(vec3 inBounds, float rndFactor) {
    float rnd4 = fract(rndFactor * 1000.0);
    vec3 norm = normalize(inBounds.xyz - vec3(0.5));
    float r = rnd4 * (1.0 - spawnBoundsSphereInnerRatio) + spawnBoundsSphereInnerRatio;
#ifndef LOCAL_SPACE
    return emitterPos + norm * r * spawnBoundsSphere;
#else
    return norm * r * spawnBoundsSphere;
#endif
}

void addInitialVelocity(inout vec3 localVelocity, vec3 inBounds) {
    localVelocity += normalize(inBounds - vec3(0.5)) * initialVelocity;
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particleUpdaterStart.js
```glsl
float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 unpack3NFloats(float src) {
    float r = fract(src);
    float g = fract(src * 256.0);
    float b = fract(src * 65536.0);
    return vec3(r, g, b);
}

vec3 tex1Dlod_lerp(highp sampler2D tex, vec2 tc, out vec3 w) {
    vec4 a = texture2D(tex, tc);
    vec4 b = texture2D(tex, tc + graphSampleSize);
    float c = fract(tc.x * graphNumSamples);

    vec3 unpackedA = unpack3NFloats(a.w);
    vec3 unpackedB = unpack3NFloats(b.w);
    w = mix(unpackedA, unpackedB, c);

    return mix(a.xyz, b.xyz, c);
}

#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)
vec4 hash41(float p) {
    vec4 p4 = fract(vec4(p) * HASHSCALE4);
    p4 += dot(p4, p4.wzxy+19.19);
    return fract(vec4((p4.x + p4.y)*p4.z, (p4.x + p4.z)*p4.y, (p4.y + p4.z)*p4.w, (p4.z + p4.w)*p4.x));
}

void main(void) {
    if (gl_FragCoord.x > numParticles) discard;

    readInput(vUv0.x);
    visMode = inShow? 1.0 : -1.0;

    vec4 rndFactor = hash41(gl_FragCoord.x + seed);

    float particleRate = rate + rateDiv * rndFactor.x;

    outLife = inLife + delta;
    float nlife = clamp(outLife / lifetime, 0.0, 1.0);

    vec3 localVelocityDiv;
    vec3 velocityDiv;
    vec3 paramDiv;
    vec3 localVelocity = tex1Dlod_lerp(internalTex0, vec2(nlife, 0), localVelocityDiv);
    vec3 velocity =      tex1Dlod_lerp(internalTex1, vec2(nlife, 0), velocityDiv);
    vec3 params =        tex1Dlod_lerp(internalTex2, vec2(nlife, 0), paramDiv);
    float rotSpeed = params.x;
    float rotSpeedDiv = paramDiv.y;

    vec3 radialParams = tex1Dlod_lerp(internalTex3, vec2(nlife, 0), paramDiv);
    float radialSpeed = radialParams.x;
    float radialSpeedDiv = radialParams.y;

    bool respawn = inLife <= 0.0 || outLife >= lifetime;
    inPos = respawn ? calcSpawnPosition(rndFactor.xyz, rndFactor.x) : inPos;
    inAngle = respawn ? mix(startAngle, startAngle2, rndFactor.x) : inAngle;

#ifndef LOCAL_SPACE
    vec3 radialVel = inPos - emitterPos;
#else
    vec3 radialVel = inPos;
#endif
    radialVel = (dot(radialVel, radialVel) > 1.0E-8) ? radialSpeed * normalize(radialVel) : vec3(0.0);
    radialVel += (radialSpeedDiv * vec3(2.0) - vec3(1.0)) * radialSpeedDivMult * rndFactor.xyz;

    localVelocity +=    (localVelocityDiv * vec3(2.0) - vec3(1.0)) * localVelocityDivMult * rndFactor.xyz;
    velocity +=         (velocityDiv * vec3(2.0) - vec3(1.0)) * velocityDivMult * rndFactor.xyz;
    rotSpeed +=         (rotSpeedDiv * 2.0 - 1.0) * rotSpeedDivMult * rndFactor.y;

    addInitialVelocity(localVelocity, rndFactor.xyz);

#ifndef LOCAL_SPACE
    outVel = emitterMatrix * localVelocity + (radialVel + velocity) * emitterScale;
#else
    outVel = (localVelocity + radialVel) / emitterScale + emitterMatrixInv * velocity;
#endif

    outPos = inPos + outVel * delta;
    outAngle = inAngle + rotSpeed * delta;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_blendAdd.js
```glsl
    dBlendModeFogFactor = 0.0;
    rgb *= saturate(gammaCorrectInput(max(a, 0.0)));
    if ((rgb.r + rgb.g + rgb.b) < 0.000001) discard;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_blendMultiply.js
```glsl
    rgb = mix(vec3(1.0), rgb, vec3(a));
    if (rgb.r + rgb.g + rgb.b > 2.99) discard;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_blendNormal.js
```glsl
    if (a < 0.01) discard;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_end.js
```glsl
    rgb = addFog(rgb);
    rgb = toneMap(rgb);
    rgb = gammaCorrectOutput(rgb);
    gl_FragColor = vec4(rgb, a);
}
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_halflambert.js
```glsl
    vec3 negNormal = normal*0.5+0.5;
    vec3 posNormal = -normal*0.5+0.5;
    negNormal *= negNormal;
    posNormal *= posNormal;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_lambert.js
```glsl
    vec3 negNormal = max(normal, vec3(0.0));
    vec3 posNormal = max(-normal, vec3(0.0));
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_lighting.js
```glsl
    vec3 light = negNormal.x*lightCube[0] + posNormal.x*lightCube[1] +
                        negNormal.y*lightCube[2] + posNormal.y*lightCube[3] +
                        negNormal.z*lightCube[4] + posNormal.z*lightCube[5];

    rgb *= light;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_normalMap.js
```glsl
    vec3 normalMap = normalize(texture2D(normalMap, vec2(texCoordsAlphaLife.x, 1.0 - texCoordsAlphaLife.y)).xyz * 2.0 - 1.0);
    vec3 normal = ParticleMat * normalMap;
```
engine\src\scene\shader-lib\chunks\particle\frag\particle_soft.js
```glsl
    float depth = getLinearScreenDepth();
    float particleDepth = vDepth;
    float depthDiff = saturate(abs(particleDepth - depth) * softening);
    a *= depthDiff;
```
engine\src\scene\shader-lib\chunks\particle\vert\particle.js
```glsl
vec3 unpack3NFloats(float src) {
    float r = fract(src);
    float g = fract(src * 256.0);
    float b = fract(src * 65536.0);
    return vec3(r, g, b);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec4 tex1Dlod_lerp(highp sampler2D tex, vec2 tc) {
    return mix( texture2D(tex,tc), texture2D(tex,tc + graphSampleSize), fract(tc.x*graphNumSamples) );
}

vec4 tex1Dlod_lerp(highp sampler2D tex, vec2 tc, out vec3 w) {
    vec4 a = texture2D(tex,tc);
    vec4 b = texture2D(tex,tc + graphSampleSize);
    float c = fract(tc.x*graphNumSamples);

    vec3 unpackedA = unpack3NFloats(a.w);
    vec3 unpackedB = unpack3NFloats(b.w);
    w = mix(unpackedA, unpackedB, c);

    return mix(a, b, c);
}

vec2 rotate(vec2 quadXY, float pRotation, out mat2 rotMatrix) {
    float c = cos(pRotation);
    float s = sin(pRotation);

    mat2 m = mat2(c, -s, s, c);
    rotMatrix = m;

    return m * quadXY;
}

vec3 billboard(vec3 InstanceCoords, vec2 quadXY) {
    #ifdef SCREEN_SPACE
        vec3 pos = vec3(-1, 0, 0) * quadXY.x + vec3(0, -1, 0) * quadXY.y;
    #else
        vec3 pos = -matrix_viewInverse[0].xyz * quadXY.x + -matrix_viewInverse[1].xyz * quadXY.y;
    #endif

    return pos;
}

vec3 customFace(vec3 InstanceCoords, vec2 quadXY) {
    vec3 pos = faceTangent * quadXY.x + faceBinorm * quadXY.y;
    return pos;
}

vec2 safeNormalize(vec2 v) {
    float l = length(v);
    return (l > 1e-06) ? v / l : v;
}

void main(void) {
    vec3 meshLocalPos = particle_vertexData.xyz;
    float id = floor(particle_vertexData.w);

    float rndFactor = fract(sin(id + 1.0 + seed));
    vec3 rndFactor3 = vec3(rndFactor, fract(rndFactor*10.0), fract(rndFactor*100.0));

    float uv = id / numParticlesPot;
    readInput(uv);

#ifdef LOCAL_SPACE
    inVel = mat3(matrix_model) * inVel;
#endif
    vec2 velocityV = safeNormalize((mat3(matrix_view) * inVel).xy); // should be removed by compiler if align/stretch is not used

    float particleLifetime = lifetime;

    if (inLife <= 0.0 || inLife > particleLifetime || !inShow) meshLocalPos = vec3(0.0);
    vec2 quadXY = meshLocalPos.xy;
    float nlife = clamp(inLife / particleLifetime, 0.0, 1.0);

    vec3 paramDiv;
    vec4 params = tex1Dlod_lerp(internalTex2, vec2(nlife, 0), paramDiv);
    float scale = params.y;
    float scaleDiv = paramDiv.x;
    float alphaDiv = paramDiv.z;

    scale += (scaleDiv * 2.0 - 1.0) * scaleDivMult * fract(rndFactor*10000.0);

#ifndef USE_MESH
    texCoordsAlphaLife = vec4(quadXY * -0.5 + 0.5, (alphaDiv * 2.0 - 1.0) * alphaDivMult * fract(rndFactor*1000.0), nlife);
#else
    texCoordsAlphaLife = vec4(particle_uv, (alphaDiv * 2.0 - 1.0) * alphaDivMult * fract(rndFactor*1000.0), nlife);
#endif

    vec3 particlePos = inPos;
    vec3 particlePosMoved = vec3(0.0);

    mat2 rotMatrix;
```
engine\src\scene\shader-lib\chunks\particle\vert\particleAnimFrameClamp.js
```glsl
    float animFrame = min(floor(texCoordsAlphaLife.w * animTexParams.y) + animTexParams.x, animTexParams.z);
```
engine\src\scene\shader-lib\chunks\particle\vert\particleAnimFrameLoop.js
```glsl
    float animFrame = floor(mod(texCoordsAlphaLife.w * animTexParams.y + animTexParams.x, animTexParams.z + 1.0));
```
engine\src\scene\shader-lib\chunks\particle\vert\particleAnimTex.js
```glsl
    float animationIndex;

    if (animTexIndexParams.y == 1.0) {
        animationIndex = floor((animTexParams.w + 1.0) * rndFactor3.z) * (animTexParams.z + 1.0);
    } else {
        animationIndex = animTexIndexParams.x * (animTexParams.z + 1.0);
    }

    float atlasX = (animationIndex + animFrame) * animTexTilesParams.x;
    float atlasY = 1.0 - floor(atlasX + 1.0) * animTexTilesParams.y;
    atlasX = fract(atlasX);

    texCoordsAlphaLife.xy *= animTexTilesParams.xy;
    texCoordsAlphaLife.xy += vec2(atlasX, atlasY);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_billboard.js
```glsl
    quadXY = rotate(quadXY, inAngle, rotMatrix);
    vec3 localPos = billboard(particlePos, quadXY);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_cpu.js
```glsl
attribute vec4 particle_vertexData;   // XYZ = world pos, W = life
attribute vec4 particle_vertexData2;  // X = angle, Y = scale, Z = alpha, W = velocity.x
attribute vec4 particle_vertexData3;  // XYZ = particle local pos, W = velocity.y
attribute float particle_vertexData4; // particle id

// type depends on useMesh property. Start with X = velocity.z, Y = particle ID and for mesh particles proceeds with Z = mesh UV.x, W = mesh UV.y
#ifndef USE_MESH
attribute vec2 particle_vertexData5;
#else
attribute vec4 particle_vertexData5;
#endif

uniform mat4 matrix_viewProjection;
uniform mat4 matrix_model;

#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif

uniform mat3 matrix_normal;
uniform mat4 matrix_viewInverse;

uniform float numParticles;
uniform float lifetime;
uniform float stretch;
uniform float seed;
uniform vec3 wrapBounds;
uniform vec3 emitterScale;
uniform vec3 faceTangent;
uniform vec3 faceBinorm;
uniform sampler2D texLifeAndSourcePosOUT;
uniform highp sampler2D internalTex0;
uniform highp sampler2D internalTex1;
uniform highp sampler2D internalTex2;
uniform vec3 emitterPos;

varying vec4 texCoordsAlphaLife;

vec2 rotate(vec2 quadXY, float pRotation, out mat2 rotMatrix)
{
    float c = cos(pRotation);
    float s = sin(pRotation);
    //vec4 rotationMatrix = vec4(c, -s, s, c);

    mat2 m = mat2(c, -s, s, c);
    rotMatrix = m;

    return m * quadXY;
}

vec3 billboard(vec3 InstanceCoords, vec2 quadXY)
{
    vec3 pos = -matrix_viewInverse[0].xyz * quadXY.x + -matrix_viewInverse[1].xyz * quadXY.y;
    return pos;
}

vec3 customFace(vec3 InstanceCoords, vec2 quadXY)
{
    vec3 pos = faceTangent * quadXY.x + faceBinorm * quadXY.y;
    return pos;
}

void main(void)
{
    vec3 particlePos = particle_vertexData.xyz;
    vec3 inPos = particlePos;
    vec3 vertPos = particle_vertexData3.xyz;
    vec3 inVel = vec3(particle_vertexData2.w, particle_vertexData3.w, particle_vertexData5.x);

    float id = floor(particle_vertexData4);
    float rndFactor = fract(sin(id + 1.0 + seed));
    vec3 rndFactor3 = vec3(rndFactor, fract(rndFactor*10.0), fract(rndFactor*100.0));

#ifdef LOCAL_SPACE
    inVel = mat3(matrix_model) * inVel;
#endif
    vec2 velocityV = normalize((mat3(matrix_view) * inVel).xy); // should be removed by compiler if align/stretch is not used

    vec2 quadXY = vertPos.xy;

#ifdef USE_MESH
    texCoordsAlphaLife = vec4(particle_vertexData5.zw, particle_vertexData2.z, particle_vertexData.w);
#else
    texCoordsAlphaLife = vec4(quadXY * -0.5 + 0.5, particle_vertexData2.z, particle_vertexData.w);
#endif
    mat2 rotMatrix;

    float inAngle = particle_vertexData2.x;
    vec3 particlePosMoved = vec3(0.0);
    vec3 meshLocalPos = particle_vertexData3.xyz;
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_cpu_end.js
```glsl
    localPos *= particle_vertexData2.y * emitterScale;
    localPos += particlePos;

    gl_Position = matrix_viewProjection * vec4(localPos, 1.0);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_customFace.js
```glsl
    quadXY = rotate(quadXY, inAngle, rotMatrix);
    vec3 localPos = customFace(particlePos, quadXY);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_end.js
```glsl
    localPos *= scale * emitterScale;
    localPos += particlePos;

    #ifdef SCREEN_SPACE
    gl_Position = vec4(localPos.x, localPos.y, 0.0, 1.0);
    #else
    gl_Position = matrix_viewProjection * vec4(localPos.xyz, 1.0);
    #endif
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_init.js
```glsl
attribute vec4 particle_vertexData; // XYZ = particle position, W = particle ID + random factor
#ifdef USE_MESH
attribute vec2 particle_uv;         // mesh UV
#endif

uniform mat4 matrix_viewProjection;
uniform mat4 matrix_model;
uniform mat3 matrix_normal;
uniform mat4 matrix_viewInverse;

#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif

uniform float numParticles, numParticlesPot;
uniform float graphSampleSize;
uniform float graphNumSamples;
uniform float stretch;
uniform vec3 wrapBounds;
uniform vec3 emitterScale, emitterPos, faceTangent, faceBinorm;
uniform float rate, rateDiv, lifetime, deltaRandomnessStatic, scaleDivMult, alphaDivMult, seed, delta;
uniform sampler2D particleTexOUT, particleTexIN;
uniform highp sampler2D internalTex0;
uniform highp sampler2D internalTex1;
uniform highp sampler2D internalTex2;

#ifndef CAMERAPLANES
#define CAMERAPLANES
uniform vec4 camera_params;
#endif

varying vec4 texCoordsAlphaLife;

vec3 inPos;
vec3 inVel;
float inAngle;
bool inShow;
float inLife;
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_localShift.js
```glsl
    particlePos = (matrix_model * vec4(particlePos, 1.0)).xyz;
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_mesh.js
```glsl
    vec3 localPos = meshLocalPos;
    localPos.xy = rotate(localPos.xy, inAngle, rotMatrix);
    localPos.yz = rotate(localPos.yz, inAngle, rotMatrix);

    billboard(particlePos, quadXY);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_normal.js
```glsl
    Normal = normalize(localPos + matrix_viewInverse[2].xyz);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_pointAlong.js
```glsl
    inAngle = atan(velocityV.x, velocityV.y); // not the fastest way, but easier to plug in; TODO: create rot matrix right from vectors

```
engine\src\scene\shader-lib\chunks\particle\vert\particle_soft.js
```glsl
    vDepth = getLinearDepth(localPos);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_stretch.js
```glsl
    vec3 moveDir = inVel * stretch;
    vec3 posPrev = particlePos - moveDir;
    posPrev += particlePosMoved;

    vec2 centerToVertexV = normalize((mat3(matrix_view) * localPos).xy);

    float interpolation = dot(-velocityV, centerToVertexV) * 0.5 + 0.5;

    particlePos = mix(particlePos, posPrev, interpolation);
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_TBN.js
```glsl
    mat3 rot3 = mat3(rotMatrix[0][0], rotMatrix[0][1], 0.0, rotMatrix[1][0], rotMatrix[1][1], 0.0, 0.0, 0.0, 1.0);
    ParticleMat = mat3(-matrix_viewInverse[0].xyz, -matrix_viewInverse[1].xyz, matrix_viewInverse[2].xyz) * rot3;
```
engine\src\scene\shader-lib\chunks\particle\vert\particle_wrap.js
```glsl
    vec3 origParticlePos = particlePos;
    particlePos -= matrix_model[3].xyz;
    particlePos = mod(particlePos, wrapBounds) - wrapBounds * 0.5;
    particlePos += matrix_model[3].xyz;
    particlePosMoved = particlePos - origParticlePos;
```
engine\src\scene\shader-lib\chunks\skybox\frag\skyboxEnv.js
```glsl
varying vec3 vViewDir;

uniform sampler2D texture_envAtlas;
uniform float mipLevel;

void main(void) {
    vec3 dir = vViewDir * vec3(-1.0, 1.0, 1.0);
    vec2 uv = toSphericalUv(normalize(dir));

    vec3 linear = $DECODE(texture2D(texture_envAtlas, mapRoughnessUv(uv, mipLevel)));

    gl_FragColor = vec4(gammaCorrectOutput(toneMap(processEnvironment(linear))), 1.0);
}
```
engine\src\scene\shader-lib\chunks\skybox\frag\skyboxHDR.js
```glsl
varying vec3 vViewDir;

uniform samplerCube texture_cubeMap;

void main(void) {
    vec3 dir=vViewDir;
    dir.x *= -1.0;

    vec3 linear = $DECODE(textureCube(texture_cubeMap, fixSeamsStatic(dir, $FIXCONST)));

    gl_FragColor = vec4(gammaCorrectOutput(toneMap(processEnvironment(linear))), 1.0);
}
```
engine\src\scene\shader-lib\chunks\skybox\vert\skybox.js
```glsl
attribute vec3 aPosition;

#ifndef VIEWMATRIX
#define VIEWMATRIX
uniform mat4 matrix_view;
#endif

uniform mat4 matrix_projectionSkybox;
uniform mat3 cubeMapRotationMatrix;

varying vec3 vViewDir;

void main(void) {
    mat4 view = matrix_view;
    view[3][0] = view[3][1] = view[3][2] = 0.0;
    gl_Position = matrix_projectionSkybox * view * vec4(aPosition, 1.0);

    // Force skybox to far Z, regardless of the clip planes on the camera
    // Subtract a tiny fudge factor to ensure floating point errors don't
    // still push pixels beyond far Z. See:
    // http://www.opengl.org/discussion_boards/showthread.php/171867-skybox-problem

    gl_Position.z = gl_Position.w - 0.00001;
    vViewDir = aPosition * cubeMapRotationMatrix;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\alphaTest.js
```glsl
uniform float alpha_ref;

void alphaTest(float a) {
    if (a < alpha_ref) discard;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\ao.js
```glsl

void getAO() {
    dAo = 1.0;

    #ifdef MAPTEXTURE
    dAo *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    dAo *= saturate(vVertexColor.$VC);
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\clearCoat.js
```glsl
#ifdef MAPFLOAT
uniform float material_clearCoat;
#endif

void getClearCoat() {
    ccSpecularity = 1.0;

    #ifdef MAPFLOAT
    ccSpecularity *= material_clearCoat;
    #endif

    #ifdef MAPTEXTURE
    ccSpecularity *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    ccSpecularity *= saturate(vVertexColor.$VC);
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\clearCoatGloss.js
```glsl
#ifdef MAPFLOAT
uniform float material_clearCoatGloss;
#endif

void getClearCoatGlossiness() {
    ccGlossiness = 1.0;

    #ifdef MAPFLOAT
    ccGlossiness *= material_clearCoatGloss;
    #endif

    #ifdef MAPTEXTURE
    ccGlossiness *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    ccGlossiness *= saturate(vVertexColor.$VC);
    #endif

    #ifdef MAPINVERT
    ccGlossiness = 1.0 - ccGlossiness;
    #endif

    ccGlossiness += 0.0000001;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\clearCoatNormal.js
```glsl
#ifdef MAPTEXTURE
uniform float material_clearCoatBumpiness;
#endif

void getClearCoatNormal() {
#ifdef MAPTEXTURE
    vec3 normalMap = unpackNormal(texture2DBias($SAMPLER, $UV, textureBias));
    normalMap = mix(vec3(0.0, 0.0, 1.0), normalMap, material_clearCoatBumpiness);
    ccNormalW = normalize(dTBN * normalMap);
#else
    ccNormalW = dVertexNormalW;
#endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\detailModes.js
```glsl
vec3 detailMode_mul(vec3 c1, vec3 c2) {
    return c1 * c2;
}

vec3 detailMode_add(vec3 c1, vec3 c2) {
    return c1 + c2;
}

// https://en.wikipedia.org/wiki/Blend_modes#Screen
vec3 detailMode_screen(vec3 c1, vec3 c2) {
    return 1.0 - (1.0 - c1)*(1.0 - c2);
}

// https://en.wikipedia.org/wiki/Blend_modes#Overlay
vec3 detailMode_overlay(vec3 c1, vec3 c2) {
    return mix(1.0 - 2.0*(1.0 - c1)*(1.0 - c2), 2.0*c1*c2, step(c1, vec3(0.5)));
}

vec3 detailMode_min(vec3 c1, vec3 c2) {
    return min(c1, c2);
}

vec3 detailMode_max(vec3 c1, vec3 c2) {
    return max(c1, c2);
}
```
engine\src\scene\shader-lib\chunks\standard\frag\diffuse.js
```glsl
#ifdef MAPCOLOR
uniform vec3 material_diffuse;
#endif

void getAlbedo() {
    dAlbedo = vec3(1.0);

#ifdef MAPCOLOR
    dAlbedo *= material_diffuse.rgb;
#endif

#ifdef MAPTEXTURE
    vec3 albedoBase = $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    dAlbedo *= addAlbedoDetail(albedoBase);
#endif

#ifdef MAPVERTEX
    dAlbedo *= gammaCorrectInput(saturate(vVertexColor.$VC));
#endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\diffuseDetailMap.js
```glsl
vec3 addAlbedoDetail(vec3 albedo) {
#ifdef MAPTEXTURE
    vec3 albedoDetail = $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    return detailMode_$DETAILMODE(albedo, albedoDetail);
#else
    return albedo;
#endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\emissive.js
```glsl
#ifdef MAPCOLOR
uniform vec3 material_emissive;
#endif

#ifdef MAPFLOAT
uniform float material_emissiveIntensity;
#endif

void getEmission() {
    dEmission = vec3(1.0);

    #ifdef MAPFLOAT
    dEmission *= material_emissiveIntensity;
    #endif

    #ifdef MAPCOLOR
    dEmission *= material_emissive;
    #endif

    #ifdef MAPTEXTURE
    dEmission *= $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    #endif

    #ifdef MAPVERTEX
    dEmission *= gammaCorrectInput(saturate(vVertexColor.$VC));
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\gloss.js
```glsl
#ifdef MAPFLOAT
uniform float material_gloss;
#endif

void getGlossiness() {
    dGlossiness = 1.0;

    #ifdef MAPFLOAT
    dGlossiness *= material_gloss;
    #endif

    #ifdef MAPTEXTURE
    dGlossiness *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    dGlossiness *= saturate(vVertexColor.$VC);
    #endif

    #ifdef MAPINVERT
    dGlossiness = 1.0 - dGlossiness;
    #endif

    dGlossiness += 0.0000001;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\iridescence.js
```glsl
#ifdef MAPFLOAT
uniform float material_iridescence;
#endif

void getIridescence() {
    float iridescence = 1.0;

    #ifdef MAPFLOAT
    iridescence *= material_iridescence;
    #endif

    #ifdef MAPTEXTURE
    iridescence *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    dIridescence = iridescence; 
}
```
engine\src\scene\shader-lib\chunks\standard\frag\iridescenceThickness.js
```glsl
uniform float material_iridescenceThicknessMax;

#ifdef MAPTEXTURE
uniform float material_iridescenceThicknessMin;
#endif

void getIridescenceThickness() {

    #ifdef MAPTEXTURE
    float blend = texture2DBias($SAMPLER, $UV, textureBias).$CH;
    float iridescenceThickness = mix(material_iridescenceThicknessMin, material_iridescenceThicknessMax, blend);
    #else
    float iridescenceThickness = material_iridescenceThicknessMax;
    #endif

    dIridescenceThickness = iridescenceThickness; 
}
```
engine\src\scene\shader-lib\chunks\standard\frag\lightmapDir.js
```glsl
uniform sampler2D texture_lightMap;
uniform sampler2D texture_dirLightMap;

void getLightMap() {
    dLightmap = $DECODE(texture2DBias(texture_lightMap, $UV, textureBias)).$CH;

    vec3 dir = texture2DBias(texture_dirLightMap, $UV, textureBias).xyz * 2.0 - 1.0;
    float dirDot = dot(dir, dir);
    dLightmapDir = (dirDot > 0.001) ? dir / sqrt(dirDot) : vec3(0.0);
}
```
engine\src\scene\shader-lib\chunks\standard\frag\lightmapSingle.js
```glsl
void getLightMap() {
    dLightmap = vec3(1.0);

    #ifdef MAPTEXTURE
    dLightmap *= $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    #endif

    #ifdef MAPVERTEX
    dLightmap *= saturate(vVertexColor.$VC);
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\metalness.js
```glsl
#ifdef MAPFLOAT
uniform float material_metalness;
#endif

void getMetalness() {
    float metalness = 1.0;

    #ifdef MAPFLOAT
    metalness *= material_metalness;
    #endif

    #ifdef MAPTEXTURE
    metalness *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    metalness *= saturate(vVertexColor.$VC);
    #endif

    dMetalness = metalness;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\normalDetailMap.js
```glsl
#ifdef MAPTEXTURE
uniform float material_normalDetailMapBumpiness;

vec3 blendNormals(vec3 n1, vec3 n2) {
    // https://blog.selfshadow.com/publications/blending-in-detail/#detail-oriented
    n1 += vec3(0, 0, 1);
    n2 *= vec3(-1, -1, 1);
    return n1 * dot(n1, n2) / n1.z - n2;
}
#endif

vec3 addNormalDetail(vec3 normalMap) {
#ifdef MAPTEXTURE
    vec3 normalDetailMap = unpackNormal(texture2DBias($SAMPLER, $UV, textureBias));
    normalDetailMap = mix(vec3(0.0, 0.0, 1.0), normalDetailMap, material_normalDetailMapBumpiness);
    return blendNormals(normalMap, normalDetailMap);
#else
    return normalMap;
#endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\normalMap.js
```glsl
#ifdef MAPTEXTURE
uniform float material_bumpiness;
#endif

void getNormal() {
#ifdef MAPTEXTURE
    vec3 normalMap = unpackNormal(texture2DBias($SAMPLER, $UV, textureBias));
    normalMap = mix(vec3(0.0, 0.0, 1.0), normalMap, material_bumpiness);
    dNormalW = normalize(dTBN * addNormalDetail(normalMap));
#else
    dNormalW = dVertexNormalW;
#endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\normalXY.js
```glsl
vec3 unpackNormal(vec4 nmap) {
    vec3 normal;
    normal.xy = nmap.wy * 2.0 - 1.0;
    normal.z = sqrt(1.0 - saturate(dot(normal.xy, normal.xy)));
    return normal;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\normalXYZ.js
```glsl
vec3 unpackNormal(vec4 nmap) {
    return nmap.xyz * 2.0 - 1.0;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\opacity.js
```glsl
#ifdef MAPFLOAT
uniform float material_opacity;
#endif

void getOpacity() {
    dAlpha = 1.0;

    #ifdef MAPFLOAT
    dAlpha *= material_opacity;
    #endif

    #ifdef MAPTEXTURE
    dAlpha *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    dAlpha *= clamp(vVertexColor.$VC, 0.0, 1.0);
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\parallax.js
```glsl
uniform float material_heightMapFactor;

void getParallax() {
    float parallaxScale = material_heightMapFactor;

    float height = texture2DBias($SAMPLER, $UV, textureBias).$CH;
    height = height * parallaxScale - parallaxScale*0.5;
    vec3 viewDirT = dViewDirW * dTBN;

    viewDirT.z += 0.42;
    dUvOffset = height * (viewDirT.xy / viewDirT.z);
}
```
engine\src\scene\shader-lib\chunks\standard\frag\sheen.js
```glsl

#ifdef MAPCOLOR
uniform vec3 material_sheen;
#endif

void getSheen() {
    vec3 sheenColor = vec3(1, 1, 1);

    #ifdef MAPCOLOR
    sheenColor *= material_sheen;
    #endif

    #ifdef MAPTEXTURE
    sheenColor *= $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    #endif

    #ifdef MAPVERTEX
    sheenColor *= saturate(vVertexColor.$VC);
    #endif

    sSpecularity = sheenColor;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\sheenGloss.js
```glsl
#ifdef MAPFLOAT
uniform float material_sheenGloss;
#endif

void getSheenGlossiness() {
    float sheenGlossiness = 1.0;

    #ifdef MAPFLOAT
    sheenGlossiness *= material_sheenGloss;
    #endif

    #ifdef MAPTEXTURE
    sheenGlossiness *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    sheenGlossiness *= saturate(vVertexColor.$VC);
    #endif

    #ifdef MAPINVERT
    sheenGlossiness = 1.0 - sheenGlossiness;
    #endif

    sheenGlossiness += 0.0000001;
    sGlossiness = sheenGlossiness;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\specular.js
```glsl

#ifdef MAPCOLOR
uniform vec3 material_specular;
#endif

void getSpecularity() {
    vec3 specularColor = vec3(1,1,1);

    #ifdef MAPCOLOR
    specularColor *= material_specular;
    #endif

    #ifdef MAPTEXTURE
    specularColor *= $DECODE(texture2DBias($SAMPLER, $UV, textureBias)).$CH;
    #endif

    #ifdef MAPVERTEX
    specularColor *= saturate(vVertexColor.$VC);
    #endif

    dSpecularity = specularColor;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\specularityFactor.js
```glsl

#ifdef MAPFLOAT
uniform float material_specularityFactor;
#endif

void getSpecularityFactor() {
    float specularityFactor = 1.0;

    #ifdef MAPFLOAT
    specularityFactor *= material_specularityFactor;
    #endif

    #ifdef MAPTEXTURE
    specularityFactor *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    specularityFactor *= saturate(vVertexColor.$VC);
    #endif

    dSpecularityFactor = specularityFactor;
}
```
engine\src\scene\shader-lib\chunks\standard\frag\textureSample.js
```glsl
vec4 texture2DSRGB(sampler2D tex, vec2 uv) {
    return gammaCorrectInput(texture2D(tex, uv));
}

vec4 texture2DSRGB(sampler2D tex, vec2 uv, float bias) {
    return gammaCorrectInput(texture2D(tex, uv, bias));
}

vec3 texture2DRGBM(sampler2D tex, vec2 uv) {
    return decodeRGBM(texture2D(tex, uv));
}

vec3 texture2DRGBM(sampler2D tex, vec2 uv, float bias) {
    return decodeRGBM(texture2D(tex, uv, bias));
}

vec3 texture2DRGBE(sampler2D tex, vec2 uv) {
    return decodeRGBM(texture2D(tex, uv));
}

vec3 texture2DRGBE(sampler2D tex, vec2 uv, float bias) {
    return decodeRGBM(texture2D(tex, uv, bias));
}
```
engine\src\scene\shader-lib\chunks\standard\frag\thickness.js
```glsl
#ifdef MAPFLOAT
uniform float material_thickness;
#endif

void getThickness() {
    dThickness = 1.0;

    #ifdef MAPFLOAT
    dThickness *= material_thickness;
    #endif

    #ifdef MAPTEXTURE
    dThickness *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    dThickness *= saturate(vVertexColor.$VC);
    #endif
}
```
engine\src\scene\shader-lib\chunks\standard\frag\transmission.js
```glsl

#ifdef MAPFLOAT
uniform float material_refraction;
#endif

void getRefraction() {
    float refraction = 1.0;

    #ifdef MAPFLOAT
    refraction = material_refraction;
    #endif

    #ifdef MAPTEXTURE
    refraction *= texture2DBias($SAMPLER, $UV, textureBias).$CH;
    #endif

    #ifdef MAPVERTEX
    refraction *= saturate(vVertexColor.$VC);
    #endif

    dTransmission = refraction;
}
```
