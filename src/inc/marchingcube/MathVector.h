#pragma once

#include <cmath>
#include <iostream>
#include <algorithm>

//for marching cubes
class iVector3 
{
public:

	iVector3() :x(0), y(0), z(0) {}
	iVector3(int x0, int y0, int z0) :x(x0), y(y0), z(z0) {}
	iVector3(const iVector3& other) :x(other.x), y(other.y), z(other.z) {}
	iVector3(int vals[3]) { x = vals[0]; y = vals[1]; z = vals[2]; }

	iVector3 operator +(const iVector3& other)const { return iVector3(x + other.x, y + other.y, z + other.z); }
	iVector3 operator -(const iVector3& other)const { return iVector3(x - other.x, y - other.y, z - other.z); }
	iVector3 operator *(const iVector3& other)const { return iVector3(x * other.x, y * other.y, z * other.z); }
	iVector3 operator /(const iVector3& other)const { return iVector3(x / other.x, y / other.y, z / other.z); }

	iVector3& operator +=(const iVector3& other) { x += other.x; y += other.y; z += other.z; return *this; }
	iVector3& operator -=(const iVector3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
	iVector3& operator *=(const iVector3& other) { x *= other.x; y *= other.y; z *= other.z; return *this; }
	iVector3& operator /=(const iVector3& other) { x /= other.x; y /= other.y; z /= other.z; return *this; }

	iVector3 operator +(int scale)const { return iVector3(x + scale, y + scale, z + scale); }
	iVector3 operator -(int scale)const { return iVector3(x - scale, y - scale, z - scale); }
	iVector3 operator *(int scale)const { return iVector3(x * scale, y * scale, z * scale); }
	iVector3 operator /(int scale)const { return iVector3(x / scale, y / scale, z / scale); }

	iVector3& operator +=(int scale) { x += scale; y += scale; z += scale; return *this; }
	iVector3& operator -=(int scale) { x -= scale; y -= scale; z -= scale; return *this; }
	iVector3& operator *=(int scale) { x *= scale; y *= scale; z *= scale; return *this; }
	iVector3& operator /=(int scale) { x /= scale; y /= scale; z /= scale; return *this; }

	bool operator <(const iVector3& other)const { return (x < other.x && y < other.y && z < other.z); }
	bool operator <=(const iVector3& other)const { return (x <= other.x && y <= other.y && z <= other.z); }
	bool operator >(const iVector3& other)const { return (x > other.x && y > other.y && z > other.z); }
	bool operator >=(const iVector3& other)const { return (x >= other.x && y >= other.y && z >= other.z); }

	int lengthSquare() const { return x * x + y * y + z * z; }
	float length() const { return sqrt((float)lengthSquare()); }

	int distanceSquare(const iVector3& other) { return iVector3(*this - other).lengthSquare(); }
	float distance(const iVector3& other) { return sqrt((float)this->distanceSquare(other)); }

	int dot(const iVector3& other) { return x * other.x + y * other.y + z * other.z; }

	void limitMin(const iVector3& minVal) { x = std::max(minVal.x, x); y = std::max(minVal.y, y); z = std::max(minVal.z, z); }
	void limitMax(const iVector3& maxVal) { x = std::min(maxVal.x, x); y = std::min(maxVal.y, y); z = std::min(maxVal.z, z); };

	int x, y, z;

};

class fVector3 
{
public:
	fVector3() :x(0.f), y(0.f), z(0.f) {}
	fVector3(float x0, float y0, float z0) :x(x0), y(y0), z(z0) {}
	fVector3(float val[3]) :x(val[0]), y(val[1]), z(val[2]) {}
	fVector3(const fVector3& other) :x(other.x), y(other.y), z(other.z) {}
	fVector3(const iVector3& other) :x((float)other.x), y((float)other.y), z((float)other.z) {}

	fVector3 operator +(const fVector3& other)const { return fVector3(x + other.x, y + other.y, z + other.z); }
	fVector3 operator -(const fVector3& other)const { return fVector3(x - other.x, y - other.y, z - other.z); }
	fVector3 operator *(const fVector3& other)const { return fVector3(x * other.x, y * other.y, z * other.z); }
	fVector3 operator /(const fVector3& other)const { return fVector3(x / other.x, y / other.y, z / other.z); }

	fVector3& operator +=(const fVector3& other) { x += other.x; y += other.y; z += other.z; return *this; }
	fVector3& operator -=(const fVector3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
	fVector3& operator *=(const fVector3& other) { x *= other.x; y *= other.y; z *= other.z; return *this; }
	fVector3& operator /=(const fVector3& other) { x /= other.x; y /= other.y; z /= other.z; return *this; }

	fVector3 operator +(float scale)const { return fVector3(x + scale, y + scale, z + scale); }
	fVector3 operator -(float scale)const { return fVector3(x - scale, y - scale, z - scale); }
	fVector3 operator *(float scale)const { return fVector3(x * scale, y * scale, z * scale); }
	fVector3 operator /(float scale)const { return fVector3(x / scale, y / scale, z / scale); }

	fVector3& operator +=(float scale) { x += scale; y += scale; z += scale; return *this; }
	fVector3& operator -=(float scale) { x -= scale; y -= scale; z -= scale; return *this; }
	fVector3& operator *=(float scale) { x *= scale; y *= scale; z *= scale; return *this; }
	fVector3& operator /=(float scale) { x /= scale; y /= scale; z /= scale; return *this; }

	bool operator <(float num) { return x < num && y < num && z < num; }
	const float& operator[](int i)const { if (i == 0) return x; else if (i == 1) return y; else return z; }

	iVector3 toIntVector() { return iVector3((int)x, (int)y, (int)z); }
	void setXYZ(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	void set(int pos, float val) { if (pos == 0)x = val; else if (pos == 1) y = val; else z = val; }

	void getArray(float val[3])const { val[0] = x; val[1] = y; val[2] = z; }
	float lengthSquare() const { return x * x + y * y + z * z; }
	float length() const { return sqrt(lengthSquare()); }

	float distanceSquare(const fVector3& other)const { return fVector3(*this - other).lengthSquare(); }
	float distance(const fVector3& other)const { return sqrt(this->distanceSquare(other)); }

	void normalize();
	fVector3 cross(const fVector3& other)const;
	float dot(const fVector3& other)const { return x * other.x + y * other.y + z * other.z; }

	// output stream.
	friend std::ostream& operator<<(std::ostream& os, const fVector3& a);

	float x, y, z;

};






