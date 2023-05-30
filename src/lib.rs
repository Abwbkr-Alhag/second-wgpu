use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use wgpu::{util::DeviceExt, BindGroup};

use cgmath::{prelude::*, Point3};
// use std::cmp::Eq;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod camera;
mod model;
mod resources;
mod texture;
mod collision;
mod ray;
mod aab;

use model::{DrawModel, Vertex};
use collision::{BVTree, Collider};
use aab::{AABRect};
use ray::Ray;

use crate::{model::DrawWireFrame, aab::WireframeRaw};

#[derive(Clone, Copy, Debug)]
pub struct Instance {
    id: usize,
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: cgmath::Vector3<f32>,
}

impl Collider for Instance {
    fn position(&self) -> Point3<f32> {
        self.position
    }
    fn scale(&self) -> cgmath::Vector3<f32> {
        self.scale
    }
    fn id(&self) -> usize {
        self.id
    }
    fn set_position(&mut self, new_position: cgmath::Point3<f32>) {
        self.position = new_position;
    }
}

impl<'a> Default for Instance {
    fn default() -> Self {
        Self {
            id: 0,
            position: cgmath::Point3 {
                x: std::f32::INFINITY,
                y: std::f32::INFINITY,
                z: std::f32::INFINITY,
            },
            rotation: cgmath::Quaternion { 
                v: cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 }, 
                s: 0.0 
            },
            scale: cgmath::Vector3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        }
    }
}

impl Instance {
    fn new(position: cgmath::Point3<f32>, rotation: cgmath::Quaternion<f32>, scale: cgmath::Vector3<f32>) -> Instance {
        static mut COUNTER: usize = 1;
        let new_instance = Self {
            id: unsafe {  COUNTER  },
            position,
            rotation,
            scale,
        };
        unsafe { COUNTER+=1; }
        return new_instance;
    }
    fn to_raw(&self) -> InstanceRaw {
        let model =
            cgmath::Matrix4::from_translation(cgmath::Vector3 {
                x: self.position.x,
                y: self.position.y,
                z: self.position.z
            }) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
            scale: self.scale.into()
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
    scale: [f32; 3],
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 25]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    light_render_pipeline: wgpu::RenderPipeline,
    debug_pipeline: wgpu::RenderPipeline,
    floor_model: model::Model,
    debug_model: model::Model,
    light_model: model::Model,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: BindGroup,
    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: BindGroup,
    camera_controller: camera::CameraController,
    projection: camera::Projection,
    collision_tree: BVTree<AABRect, Instance>,
    instance_buffer: wgpu::Buffer,
    tree_debug_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> State {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb && f.describe().components > 3)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        
        let mut rng = rand::thread_rng();
        use rand::Rng;
        const NUM_OF_INSTANCES: u32 = 25;
        let instances: Vec<Instance> = (0..NUM_OF_INSTANCES).map(|_| {
            Instance::new(
                cgmath::Point3 {
                    x: rng.gen_range(-100.0..100.0),
                    y: rng.gen_range(-100.0..100.0),
                    z: rng.gen_range(-100.0..100.0),
                },
                {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                },
                cgmath::Vector3 {
                    x: rng.gen_range(0.25..5.0),
                    y: rng.gen_range(0.25..5.0),
                    z: rng.gen_range(0.25..5.0),
                },
            )
        }).collect();

        let instance_data: Vec<InstanceRaw> = instances.iter().map(|inst| Instance::to_raw(inst)).collect();
        
        let mut collision_tree: BVTree<AABRect, Instance> = BVTree::new();
        for inst in instances {
            collision_tree.insert(inst);
        }

        println!("{:?}", collision_tree.get_possible_pairs());

        let tree_debug_data: Vec<WireframeRaw> = collision_tree.get_branches().iter().map(|branch| AABRect::to_raw(branch)).collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let tree_debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tree Debug Buffer"),
            contents: bytemuck::cast_slice(&tree_debug_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
            
        let floor_model = resources::load_model(
            "cube/cube.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        )
        .await
        .unwrap();

        let debug_model = resources::load_wireframe_model(
            "wireframe/untitled.obj",
            &device,
        )
        .await
        .unwrap();

        let light_model = resources::load_model(
            "cube/cube.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        )
        .await
        .unwrap();

        let light_uniform = LightUniform {
            position: [2.0, 1.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(60.0), 0.1, 2000.0);
        let camera_controller = camera::CameraController::new(40.0, 2.0);

        let mut camera_uniform = camera::CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
                false,
            )
        };

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline = {
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &texture_bind_group_layout,
                        &camera_bind_group_layout,
                        &light_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };

            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                false,
            )
        };

        let debug_pipeline = {
            let debug_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Debug Pipeline Layout"),
                    bind_group_layouts: &[
                        &camera_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Wireframe Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("wireframe.wgsl").into()),
            };

            create_render_pipeline(
                &device,
                &debug_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::WireframeModelVertex::desc(), WireframeRaw::desc()],
                shader,
                true
            )
        };

        println!("{:?}", collision_tree);
        State {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            light_render_pipeline,
            debug_pipeline,
            floor_model,
            debug_model,
            light_model,
            light_uniform,
            light_buffer,
            light_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            projection,
            collision_tree,
            instance_buffer,
            tree_debug_buffer,
            depth_texture,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent, mouse_pos: (f64, f64)) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.left_mouse_pressed = *state == ElementState::Pressed;
                true
            },
            WindowEvent::MouseInput { button: MouseButton::Right, state: ElementState::Pressed, .. } => {
                self.right_mouse_pressed = true;
                let ray = Ray::screen_coord_to_ray(&self.camera, &self.projection, &self.config, mouse_pos.0, mouse_pos.1);
                let selected = ray.find_closest_object(&self.collision_tree);
                if selected.is_some() {
                    println!("{:?}", selected.unwrap());
                }
                true
            },
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        // Update the Collision Tree
        self.collision_tree.solve_collisions();
        let instance_data = self.collision_tree.iter().map(Instance::to_raw).collect::<Vec<InstanceRaw>>();
        self.instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let tree_debug_data: Vec<WireframeRaw> = self.collision_tree.get_branches().iter().map(|branch| AABRect::to_raw(branch)).collect();
        self.tree_debug_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tree Debug Buffer"),
            contents: bytemuck::cast_slice(&tree_debug_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        //Update the Camera
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the Light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position = (cgmath::Quaternion::from_axis_angle(
            (0.0, 1.0, 0.0).into(),
            cgmath::Deg(60.0 * dt.as_secs_f32()),
        ) * old_position)
            .into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        use crate::model::DrawLight;
        render_pass.set_pipeline(&self.light_render_pipeline);
        render_pass.draw_light_model(
            &self.light_model,
            &self.camera_bind_group,
            &self.light_bind_group,
        );
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw_model_instanced(
            &self.floor_model,
            0..self.collision_tree.size() as u32,
            &self.camera_bind_group,
            &self.light_bind_group,
        );
        render_pass.set_pipeline(&self.debug_pipeline);
        render_pass.set_vertex_buffer(1, self.tree_debug_buffer.slice(..));
        render_pass.draw_wireframe_model_instanced(
            &self.debug_model, 
            0..self.collision_tree.len_branches() as u32, 
            &self.camera_bind_group
        );
        drop(render_pass);

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    debug: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: if debug { None } else { Some(wgpu::Face::Back) },
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;
    let mut last_render_time = instant::Instant::now();
    let mut mouse_pos = (0.0, 0.0);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => state.window().request_redraw(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.left_mouse_pressed {
                    state.camera_controller.process_mouse(delta.0, delta.1)
                } else if state.right_mouse_pressed {
                    // TODO:
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() && !state.input(event, mouse_pos) => match event {
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                },
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                },
                WindowEvent::CursorMoved { position, .. } => {
                    mouse_pos.0 = position.x;
                    mouse_pos.1 = position.y;
                },
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = instant::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
    });
}
