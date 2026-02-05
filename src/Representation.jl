module Representation
using CairoMakie, Agents

export video_simulation

function video_simulation(model, file_name, sim_title, agent_shape, agent_size, sim_framerate, sim_frames, color1, color2)
    # Visualización
    groupcolor(a) = a.status == false ? color1 : color2

    abmvideo(
        file_name, model;
        agent_color = groupcolor, 
        agent_marker = agent_shape, 
        agent_size = agent_size,
        framerate = sim_framerate, 
        frames = sim_frames,
        title = sim_title
    )

    println("Simulación finalizada.")
    end

end