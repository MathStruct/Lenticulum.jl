using SHA
using Base64

function alphanumeric_hash(tikz_code::String)::String
    # Create SHA256 hash and convert to alphanumeric string
    hash_bytes = sha256(tikz_code)
    # Convert to base64 and remove non-alphanumeric characters
    hash_b64 = base64encode(hash_bytes)
    # Keep only alphanumeric characters and take first 16 chars for reasonable length
    alphanumeric = replace(hash_b64, r"[^A-Za-z0-9]" => "")
    return alphanumeric[1:min(16, length(alphanumeric))]
end

function extract_tikz_from_html_comment(content::String)::Vector{String}
    # Find all TikZ code blocks inside HTML comments
    tikz_blocks = String[]
    
    # Pattern to match HTML comments containing TikZ code
    comment_pattern = r"<!--\s*(.*?)\s*-->"s
    
    for match in eachmatch(comment_pattern, content)
        comment_content = match.captures[1]
        # Check if this comment contains TikZ code
        if occursin("tikzcd", comment_content) || occursin("tikzpicture", comment_content)
            push!(tikz_blocks, strip(comment_content))
        end
    end
    
    return tikz_blocks
end

function tikz_to_svg(tikz_code::String, output_path::String)::Bool
    # Create temporary directory
    temp_dir = mktempdir()
    
    try
        # Create LaTeX document with TikZ code
        latex_content = """
        \\documentclass[tikz,border=2pt]{standalone}
        \\usepackage{tikz}
        \\usepackage{tikz-cd}
        \\usetikzlibrary{cd}
        \\begin{document}
        $tikz_code
        \\end{document}
        """
        
        # Write LaTeX file
        tex_file = joinpath(temp_dir, "tikz_diagram.tex")
        write(tex_file, latex_content)
        
        # Compile to PDF
        pdf_file = joinpath(temp_dir, "tikz_diagram.pdf")
        latex_cmd = `pdflatex -output-directory=$temp_dir -interaction=nonstopmode $tex_file`
        
        result = run(latex_cmd; wait=true)
        if !success(result)
            @error "LaTeX compilation failed"
            return false
        end
        
        # Convert PDF to SVG using pdf2svg
        svg_cmd = `pdf2svg $pdf_file $output_path`
        result = run(svg_cmd; wait=true)
        
        if !success(result)
            @error "PDF to SVG conversion failed"
            return false
        end
        
        return true
        
    catch e
        @error "Error in TikZ to SVG conversion: $e"
        return false
    finally
        # Clean up temporary directory
        rm(temp_dir; recursive=true, force=true)
    end
end

function process_markdown_file(file_path::String)
    # Read the markdown file
    if !isfile(file_path)
        @error "File not found: $file_path"
        return
    end
    
    content = read(file_path, String)
    original_content = content
    
    # Extract base filename without extension
    base_name = splitext(basename(file_path))[1]
    
    # Create SVGs directory if it doesn't exist
    svg_dir = "SVGs"
    if !isdir(svg_dir)
        mkdir(svg_dir)
    end
    
    # Extract TikZ blocks
    tikz_blocks = extract_tikz_from_html_comment(content)
    
    if isempty(tikz_blocks)
        @info "No TikZ blocks found in $file_path"
        return
    end
    
    @info "Found $(length(tikz_blocks)) TikZ block(s) in $file_path"
    
    # Process each TikZ block
    for tikz_code in tikz_blocks
        # Generate hash for this TikZ code
        hash_str = alphanumeric_hash(tikz_code)
        svg_filename = "$(base_name)-$(hash_str).svg"
        svg_path = joinpath(svg_dir, svg_filename)
        
        # Check if SVG is already embedded in the markdown
        svg_embed = "![TikZ Diagram]($svg_path)"
        if occursin(svg_embed, content)
            @info "SVG already embedded for hash $hash_str, skipping"
            continue
        end
        
        # Generate SVG if it doesn't exist
        if !isfile(svg_path)
            @info "Generating SVG: $svg_path"
            if !tikz_to_svg(tikz_code, svg_path)
                @error "Failed to generate SVG for hash $hash_str"
                continue
            end
        else
            @info "SVG already exists: $svg_path"
        end
        
        # Find the HTML comment containing this TikZ code and replace it
        comment_pattern = "<!--\\s*" * replace(tikz_code, r"[()[\]{}*+?.\\^$|]" => s"\\&") * "\\s*-->"
        
        # Create replacement text: keep comment but add SVG below it
        replacement = "<!--\n$tikz_code\n-->\n\n$svg_embed"
        
        # Replace the comment with comment + SVG
        content = replace(content, Regex(comment_pattern, "s") => replacement)
    end
    
    # Write back to file only if content changed
    if content != original_content
        write(file_path, content)
        @info "Updated $file_path with SVG embeddings"
    else
        @info "No changes needed for $file_path"
    end
end

function process_directory(dir_path::String = ".")
    # Process all .md files in the directory
    for file in readdir(dir_path)
        if endswith(file, ".md")
            file_path = joinpath(dir_path, file)
            @info "Processing: $file_path"
            process_markdown_file(file_path)
        end
    end
end

# Main execution
if length(ARGS) > 0
    # Process specific file or directory
    target = ARGS[1]
    if isfile(target) && endswith(target, ".md")
        process_markdown_file(target)
    elseif isdir(target)
        process_directory(target)
    else
        @error "Invalid target: $target (must be .md file or directory)"
    end
else
    # Process current directory
    process_directory()
end
