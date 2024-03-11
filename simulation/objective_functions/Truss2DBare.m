function varargout = Truss2DBare(varargin)
P1 = varargin{1};
P2 = varargin{2};
P3 = varargin{3};
L = varargin{4};
A = varargin{5};
E = varargin{6};
varargout{1}=RunFEM(P1, P2, P3, L, A, E);

%
%   Get joints data
%
function joints=GetJoints(length)
joints = [
    0,0;
    length,0;
    length*2,0;
    0,length;
    length,length;
    length*2,length
    ];

function members=GetMembers(area)
members = [
    1,2,area,0,0,0,0;
    2,3,area,0,0,0,0;
    4,5,area,0,0,0,0;
    5,6,area,0,0,0,0;
    2,5,area,0,0,0,0;
    3,6,area,0,0,0,0;
    1,5,area,0,0,0,0;
    2,6,area,0,0,0,0;
    2,4,area,0,0,0,0;
    3,5,area,0,0,0,0;
    ];

function supports=GetSupports()
supports = [1,1,1;4,1,1];

function loads=GetLoads(P1, P2, P3)
loads=[2,0,-P1;3,P2,-P3];

function E=GetYoungModulus()
E=100000;

% --- Data preprocessing?
function ProcessData(P1, P2, P3, L, A, E)
global members joints angles supports loads scale;
%
%  Joints
%
joints=GetJoints(L);
angles=zeros(size(joints,1),8);
ap=1.3;
if ~isempty(joints) && size(joints,2)==2
    x=joints(:,1);
    y=joints(:,2);
    xmin=min(x);
    xmax=max(x);
    dx=(xmax-xmin)/2;
    ymin=min(y);
    ymax=max(y);
    dy=(ymax-ymin)/2;
    if dx==0
        dx=1;
    end
    if dy==0
        dy=1;
    end
end
%
%  Members
%
members=GetMembers(A);

if ~isempty(members)
    for i=1:size(members,1)
        j1=members(i,1);
        j2=members(i,2);
        Lx=joints(j2,1)-joints(j1,1);
        Ly=joints(j2,2)-joints(j1,2);
        L=sqrt(Lx*Lx+Ly*Ly);
        if L==0
            disp(['*** Error *** ==>  element ' num2str(i) ,' has zero length']);
            return;
        else
            members(i,4)=E;
            members(i,5)=L;
            members(i,6)=Lx/L;
            members(i,7)=Ly/L;
        end
        ang=atan2d(Ly,Lx);
        q=round(ang/45)+1;
        q=q+8*(q<1);
        angles(j1,q)=angles(j1,q)+1;
        q=q-4;
        q=q+8*(q<1);
        angles(j2,q)=angles(j2,q)+1;
    end
end

sc=sqrt(dx^2+dy^2);
scale=sc;
supports=[];
applySupport(1,sc,0);
applySupport(4,sc,0);

loads=GetLoads(P1, P2, P3);


% --------------------------------------------------------------------
function displacement=RunFEM(P1, P2, P3, L, A, E)
global members joints angles supports loads scale;

ProcessData(P1, P2, P3, L, A, E)

for i=1:size(joints,1)
    if isempty(find(members(:,1:2)==i, 1))
        applySupport(i,scale,0);
        disp(['*** Warning *** ==>  joint ', num2str(i) ,' is not connected to any element and restarined by program']);
    end
end

nj=size(joints,1);
nm=size(members,1);
MemberForces=zeros(nm,1);
F=zeros(2*nj,1);
delta=F;
%
%  Global Stiffness Matrix before applying boundary conditions
%
GlobalK=zeros(2*nj,2*nj);
for m=1:nm
    member=members(m,:);
    i=member(1);
    j=member(2);
    elementK=MemberK(member);
    p=[2*i-1:2*i,2*j-1:2*j];
    GlobalK(p,p)=GlobalK(p,p)+elementK;
end
%
cv=10000;
unL='m';
unF='KN';
%
%  Apply loads
%
for i=1:size(loads,1)
    j=loads(i,1);
    F(2*j-1)=loads(i,2);
    F(2*j)=loads(i,3);
end
%
%   Applying Bounday Conditions
%
restDof=[];
freeDof=1:2*nj;
for i=1:size(supports,1)
    j=supports(i,1);
    if supports(i,2)==1
        restDof=[restDof 2*j-1];
    end
    if supports(i,3)==1
        restDof=[restDof 2*j];
    end
end
freeDof(restDof)=[];
reducedK = GlobalK(freeDof,freeDof);
d=diag(reducedK);
if any(d<1e-16)
    uiwait(errordlg('The structure is unstable','Error','modal'));    
    return;
end    
D=sqrt(diag(1./d));
scaledK=D'*reducedK*D;
Kp=GlobalK(restDof,freeDof);
reducedLoad=F(freeDof);
if det(scaledK)<1e-16
    uiwait(errordlg('The structure is unstable','Error','modal'));    
    return;
else
    upr=scaledK\(D*reducedLoad);
    uu=D*upr;
    Reactions=Kp*uu;
    uu=uu/E;
    delta(freeDof)=uu;
    u=reshape(delta,[2,nj])';
end
%
uref=u*cv;
st=char('',[blanks(1),'*** Dispalcements(mm) ***'],'',[blanks(5) 'Joint     X-Dir    Y-Dir'],[blanks(3),'-'*ones(1,30)]);
for i=1:nj
    st=char(st,sprintf('%9d%11.4f%10.4f',i,uref(i,1),uref(i,2)));
end
displacement=uref(3,2);
stD=st;
%
%MemberForces
%
st=char('',[blanks(1),'*** Member Forces(' unF ') ***'],'');
st=char(st,'   Element   From   To     Force',[blanks(1),'-'*ones(1,45)]);
for i=1:nm
    member=members(i,:);
    A=member(3);
    E=member(4);
    L=member(5);
    n=member(6:7);
    %[i A L  n]
    mf=A*E/L*[-n n]*[u(members(i,1),:) u(members(i,2),:)]';
    MemberForces(i)=mf;
    st=char(st,sprintf('%9d%7d%6d%11.4f',i,members(i,1),members(i,2),mf));
end
stM=st;
st=char('',[blanks(1),'*** Reactions ***'],'',[blanks(5) 'Joint    Fx(' unF ')    Fy(' unF ')'],[blanks(3),'-'*ones(1,30)],'');
n=0;
for i=1:nj
    sr=find(supports(:,1)==i);
    if ~isempty(sr)
        j=sr(1);
        rj=find(loads(:,1)==i,1);
        fxp=0;
        fyp=0;
        if ~isempty(rj)
            fxp=loads(rj,2);
            fyp=loads(rj,3);
        end
        if supports(j,2)==1
            if supports(j,3)==1
                n=n+2;
                Reactions(n-1)=Reactions(n-1)-fxp;              
                Reactions(n)=Reactions(n)-fyp;              
                st=char(st,sprintf('%9d%11.4f%11.4f',i,(Reactions(n-1)),Reactions(n)));
            else
                n=n+1;
                Reactions(n)=Reactions(n)-fxp;              
                st=char(st,sprintf('%9d%11.4f',i,Reactions(n)));
            end
        else
            n=n+1;
            Reactions(n)=Reactions(n)-fyp;              
            st=char(st,sprintf('%9d%22.4f',i,Reactions(n)));
        end
    end
end
disp(st)
stR=st;

function applySupport(j,sc,cd)
global  joints angles supports ;
x0=joints(j,1);
y0=joints(j,2);
n=0;
ang=angles(j,:);
k=20;
switch cd
    case 0
        supports=[supports ; j 1 1];
    case 1
        supports=[supports ; j 0 1];
    case 2
        supports=[supports ; j 1 0];
end
if cd<2
    if ang(3)> ang(7)
        angles(j,7)=angles(j,7)+1;
    elseif ang(3)==ang(7)
        if ang(2)+ang(4)>=ang(6)+ang(8)
            angles(j,7)=angles(j,7)+1;
        else
            angles(j,3)=angles(j,3)+1;
        end
    else
        angles(j,3)=angles(j,3)+1;
    end
end

if cd==2
    if ang(1)> ang(5)
        angles(j,5)=angles(j,5)+1;
    else
        angles(j,1)=angles(j,1)+1;
    end
end

function kel = MemberK(member)
    A=member(3);
    E=member(4);
    L=member(5);
    n=member(6:7);
    k=A/L*(n'*n);
    kel=[k -k ; -k k];
