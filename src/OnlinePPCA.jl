# Streaming soft-assignment PPCA mixture for the 2D comparison.
# In two dimensions rank-one PPCA + isotropic noise spans any positive definite
# covariance. This validates online statistics, not high-dimensional codec savings.
mutable struct OnlinePPCAComponent
    mass::Float64
    mx::Float64
    my::Float64
    xx::Float64
    xy::Float64
    yy::Float64
end
OnlinePPCAComponent(x,w)=OnlinePPCAComponent(w,x[1],x[2],0.,0.,0.)
function ppca_geometry(c; floor=1e-5)
    a=c.xx/c.mass; b=c.xy/c.mass; d=c.yy/c.mass
    gap=hypot(a-d,2b)
    large=max(floor,(a+d+gap)/2); noise=max(floor,(a+d-gap)/2)
    angle=.5atan(2b,a-d)
    (large=large,noise=noise,ux=cos(angle),uy=sin(angle))
end
function ppca_add!(c,x,w)
    w==0 && return
    total=c.mass+w; dx=x[1]-c.mx; dy=x[2]-c.my
    factor=c.mass*w/total
    c.xx+=factor*dx*dx; c.xy+=factor*dx*dy; c.yy+=factor*dy*dy
    c.mx+=w/total*dx; c.my+=w/total*dy; c.mass=total
end
function ppca_ingest!(cs,x,w,cap,radius)
    # Novelty seeding only until the fixed component budget fills.
    if isempty(cs) || (length(cs)<cap && minimum((x[1]-c.mx)^2+(x[2]-c.my)^2 for c in cs)>radius^2)
        push!(cs,OnlinePPCAComponent(x,w)); return
    end
    scores=Float64[]
    for c in cs
        g=ppca_geometry(c;floor=max(1e-5,radius^2*.05))
        dx=x[1]-c.mx; dy=x[2]-c.my
        u=dx*g.ux+dy*g.uy; v=-dx*g.uy+dy*g.ux
        push!(scores,log(c.mass)-.5*(log(g.large*g.noise)+u*u/g.large+v*v/g.noise))
    end
    scores .= exp.(scores.-maximum(scores)); scores ./= sum(scores)
    for (c,r) in zip(cs,scores); ppca_add!(c,x,w*r); end
end
function ppca_samples(c,box=(0.,1.,0.,1.))
    g=ppca_geometry(c)
    points=Tuple{Float64,Float64}[]
    # Equal-weight deterministic Gaussian quadrature, conditioned approximately
    # on unit-square support by discarding outside nodes and renormalizing.
    for radial in 1:8, angular in 1:8
        r=sqrt(-2log((radial-.5)/8)); theta=2pi*(angular-.5)/8
        u=sqrt(g.large)*r*cos(theta); v=sqrt(g.noise)*r*sin(theta)
        x=c.mx+u*g.ux-v*g.uy; y=c.my+u*g.uy+v*g.ux
        box[1]<=x<=box[2] && box[3]<=y<=box[4] && push!(points,(x,y))
    end
    isempty(points) && push!(points,(clamp(c.mx,box[1],box[2]),clamp(c.my,box[3],box[4])))
    points
end
