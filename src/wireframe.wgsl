// Vertex Shader
struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(0)@binding(0)
var<uniform> camera: Camera;

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) scale_vector: vec3<f32>,
};

// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let scaled_position = vec3<f32>(
        model.position.x * instance.scale_vector.x,
        model.position.y * instance.scale_vector.y,
        model.position.z * instance.scale_vector.z,
    );

    let world_position = model_matrix * vec4<f32>(scaled_position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    return out;
}


// Wireframe needs to be a white color
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
