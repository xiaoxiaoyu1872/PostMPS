#include "MathVector.h"

const float EPSILON_MATH = (float)1.0e-7;

void fVector3::normalize()
{
	float len = length();
	if (len > EPSILON_MATH)
	{
		float fac = 1.f / len;
		this->x *= fac;
		this->y *= fac;
		this->z *= fac;
	}
}

fVector3 fVector3::cross(const fVector3 & other) const
{
	fVector3 res;
	res.x = y * other.z - z * other.y;
	res.y = z * other.x - x * other.z;
	res.z = x * other.y - y * other.x;
	return res;
}
