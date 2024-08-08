function model = LoadModel()
%LOADMODEL Loads the .stl file.
%   The stl file in in milimeters so we downscale by 1e-3.
if evalin('base', "exist('model','var') == 0")
    model = createpde('structural','static-planestress');
    
    % Import STL
    stlFilename = 'Stiffener2D.stl';
    importGeometry(model,stlFilename);
    scale(model.Geometry, 1e-3);
    assignin('base','model',model);
else
    model = evalin('base','model');
end

