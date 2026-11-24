using Dates: Dates, now, UTC

"""Print the Julia, machine, and checkout details for a benchmark study."""
function provenance(io = stdout)
    git = `git -C $(dirname(@__DIR__))`
    commit = readchomp(`$git rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`$git status --porcelain`))
    cpu = first(Sys.cpu_info()).model
    println(io, "Julia ", VERSION, " | threads: ", Threads.nthreads(:default),
            " default, ", Threads.nthreads(:interactive), " interactive | CPU: ",
            cpu, " | OS: ", Sys.KERNEL,
            " | commit: ", commit, dirty ? "-dirty" : "", " | UTC: ", now(UTC))
    return nothing
end
