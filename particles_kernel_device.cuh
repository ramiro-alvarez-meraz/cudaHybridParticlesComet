/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* CUDA Device code for particle simulation.
*/

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "nvMath.h"
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

cudaTextureObject_t currentTex;
cudaTextureObject_t magFieldTex;
cudaTextureObject_t elecFieldTex;
// simulation parameters
__constant__ SimParams params;

// look up in 3D current texture
__device__
float3 current3D(float3 p, cudaTextureObject_t currentTex)
{
    float4 n = tex3D<float4>(currentTex, p.x, p.y, p.z);
    return make_float3(n.x, n.y, n.z);
}

// look up in 3D magnetic field texture
__device__
float3 mag3D(float3 m, cudaTextureObject_t magFieldTex)
{
    float4 n = tex3D<float4>(magFieldTex, m.x, m.y, m.z);
    return make_float3(n.x, n.y, n.z);
}

// look up in 3D magnetic field texture
__device__
float3 magGrad3D(float3 m, cudaTextureObject_t magFieldTex)
{
    float4 n = tex3D<float4>(magFieldTex, m.x, m.y, m.z);
    float4 r = {n.z-n.y, n.x-n.z, n.y-n.x};
    return make_float3(r.x, r.y, r.z);
}

// look up in 3D electric field texture
__device__
float3 elec3D(float3 e, cudaTextureObject_t elecFieldTex)
{
    float4 n = tex3D<float4>(elecFieldTex, e.x, e.y, e.z);
    return make_float3(n.x, n.y, n.z);
}

// integrate particle attributes
struct integrate_functor
{
    float deltaTime;
    cudaTextureObject_t currentTex;
    cudaTextureObject_t magFieldTex;
    cudaTextureObject_t elecFieldTex;

    __host__ __device__
    integrate_functor(float delta_time, cudaTextureObject_t current_Tex, cudaTextureObject_t magField_Tex, cudaTextureObject_t elecField_Tex) : deltaTime(delta_time), currentTex(current_Tex), magFieldTex(magField_Tex), elecFieldTex(elecField_Tex) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<2>(t);
        volatile float4 velData = thrust::get<3>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float dist = length(pos);
        float h = 1.f;		// cell size,

        // current density
        float3 curr = {0.f, 0.f, 0.f};
        float3 J = current3D(curr, currentTex);
        //float3 J = vel;

        // magnetic field
        float3 mag = {0.f, 0.f, 0.f};
        float3 magField = mag3D(mag, magFieldTex);

        // gradient of magnetic field
        float3 curlB = {(magField.z-magField.y)/h, (magField.x-magField.z)/h, (magField.y-magField.x)/h};
        float densI = 1.f/dist;					// inversely to distance

        float mu = 1.25663706212e-6f;			//magnetic permeability in vacuum [H/m]
        float eta = 4.f*3.1416f*1.e-7f;			//resistivity [Ohm * m]
        float m = 1.6726219e-27f;				//proton mass [kg]
        float q = 1.602176634e-19f;				//electron charge [Coulomb]
        float k = 1.38064852e-23f;				//boltzman constant

        //float3 grad = magGradField;
        float dens_e = densI;
        //float3 divP = {dens_e, dens_e, dens_e};

        float3 divP = {0.f, 0.f, 0.f};

        // components of the magnetic field
        float3 E_conv = cross(J, magField)/densI;
        float3 E_hall = cross(curlB, magField)/(densI*mu);
        float3 E_amb = divP/densI;
        float3 E_O = curlB*(eta/mu);

        float3 E_nO = (-E_conv + E_hall - E_amb);
        float3 E = E_nO + E_O;

        // curl of electric field
        float3 curlE_nO = {(E_nO.z-E_nO.y)/h, (E_nO.x-E_nO.z)/h, (E_nO.y-E_nO.x)/h};
        float3 curlE_O = {(E_O.z-E_O.y)/h, (E_O.x-E_O.z)/h, (E_O.y-E_O.x)/h};

        // the magnetic field is extracted from Faraday laws
        float3 B_nO = (curlE_nO) * (- deltaTime);
        float3 B_O = (curlE_O) * (- deltaTime);
        float3 B = B_nO + B_O;

        ///////////////////////
        // solving motion equations

        // apply magnetic field on velocity particles
        vel += (q/m) * cross(vel, B) * deltaTime;

        // apply electric field on velocity particles
        vel += (q/m) * E * deltaTime;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // particle outlet
        //vel += {0.01f*(pos.x/powf(dist,1)), 0.01f*(pos.y/powf(dist,1)), 0.01f*(pos.z/powf(dist,1))};

        // particle push on -z direction
        //vel += {0.f, 0.f, -0.0001f};
        //vel += EnO * deltaTime;

        // update particle age
        float age = posData.w;
        float lifetime = velData.w;

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, age);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

struct calcDepth_functor
{
    float3 sortVector;

    __host__ __device__
    calcDepth_functor(float3 sort_vector) : sortVector(sort_vector) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        volatile float4 p = thrust::get<0>(t);
        float key = -dot(make_float3(p.x, p.y, p.z), sortVector); // project onto sort vector
        thrust::get<1>(t) = key;
    }
};

#endif
