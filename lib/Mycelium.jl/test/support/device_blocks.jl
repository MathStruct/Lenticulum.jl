@eval DeviceMatrix begin
    using Mycelium
    # Check the raw block payload, including shared arrays. Flags and labels are structural.
    device_block_payload(b::Mycelium.TrivialBlock) = ()
    device_block_payload(b::Mycelium.PointBlock) = device_payload(b.X)
    device_block_payload(b::Mycelium.GaussianBlock) = device_payload((b.η, b.Λ))
    device_block_payload(b::Mycelium.CategoricalBlock) = device_payload(b.logp)
    device_block_payload(b::Mycelium.MixtureBlock) = device_payload((b.η, b.Λ, b.logw))
    device_block_payload(b::Mycelium.ComponentMixtureBlock) = device_payload((b.η, b.Λs, b.logw))
    device_block_payload(b::Mycelium.VectorBlock) = device_payload(b.beliefs)
    device_agrees(a::Mycelium.BeliefBlock, b::Mycelium.BeliefBlock, rtol) =
        device_agrees(Mycelium.replicas(a), Mycelium.replicas(b), rtol)

    device_payload(b::Mycelium.BeliefBlock) = device_block_payload(b)
    constant_likelihood(b::Mycelium.BeliefBlock, atol) =
        constant_likelihood(Mycelium.replicas(b), atol)
end
