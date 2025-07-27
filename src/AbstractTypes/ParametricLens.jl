abstract type ParametricLens{P, S, T} end

# Primary interface - these must be implemented
function get end
function set end
