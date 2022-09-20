using PSRDADA, Logging
include("../src/RFIKiller.jl")

const IN_KEY = 0xb0ba
const OUT_KEY = 0xcafe
const CHANNELS = 2048
const SAMPLES = 16384
const DTYPE = UInt16

function main()
    in_client = client_connect(IN_KEY)
    out_client = client_connect(OUT_KEY)
    @info "Connected to DADA buffers"

    # Only one header to relay
    with_read_iter(in_client; type=:header) do rb
        with_write_iter(out_client; type=:header) do wb
            next(wb) .= next(rb)
        end
    end

    @info "Relayed header info, starting RFI processing"

    n = 0

    with_read_iter(in_client; type=:data) do rb
        with_write_iter(out_client; type=:data) do wb
            for raw_spectra in rb
                spectra = reshape(reinterpret(DTYPE, raw_spectra), (CHANNELS, SAMPLES))
                spectra_floats = Float32.(spectra)
                RFIKiller.kill_rfi!(spectra_floats)
                next(wb) .= reinterpret(UInt8, vec(spectra_floats))
                n += 1
                @info "Processed $n chunks"
            end
        end
    end

    @info "Shutting down"
    cleanup(in_client)
    cleanup(out_client)
end

main()