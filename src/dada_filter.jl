using PSRDADA, Logging, ArgParse

const DTYPE = Float32

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "in_key"
        help = "Input PSRDADA key (in hex, sans leading 0x)"
        required = true
        "out_key"
        help = "Input PSRDADA key (in hex, sans leading 0x)"
        required = true
        "--channels"
        help = "Number of frequency channels"
        arg_type = Int
        default = 2048
        "--samples"
        help = "Number of time samples"
        arg_type = Int
        default = 65536
    end

    return parse_args(s)
end

function julia_main()::Cint

    parsed_args = parse_commandline()

    in_key = parse(UInt16, parsed_args["in_key"]; base=16)
    out_key = parse(UInt16, parsed_args["in_key"]; base=16)
    channels = parsed_args["channels"]
    samples = parsed_args["samples"]

    in_client = client_connect(in_key)
    out_client = client_connect(out_key)
    @info "Connected to DADA buffers"

    # Only one header to relay
    @info "Waiting for incoming header to relay"
    with_read_iter(in_client; type=:header) do rb
        with_write_iter(out_client; type=:header) do wb
            next(wb) .= next(rb)
        end
    end

    @info "Relayed header info, starting RFI processing"

    n = 0

    # Construct mask
    mask = ones(Bool, channels, samples)

    with_read_iter(in_client; type=:data) do rb
        with_write_iter(out_client; type=:data) do wb
            while true
                raw_spectra = next(rb)
                if isnothing(raw_spectra)
                    break
                end
                spectra = reshape(reinterpret(DTYPE, raw_spectra), (channels, samples))
                RFIKiller.kill_rfi!(spectra, mask)
                next(wb) .= reinterpret(UInt8, vec(spectra))
                n += 1
                @info "Processed $n chunks"
            end
        end
    end

    @info "Shutting down"
    cleanup(in_client)
    cleanup(out_client)
    return 0 # if things finished successfully
end