#date: 2025-07-21T16:41:29Z
#url: https://api.github.com/gists/a918958f4c1b98aef83618111fe54e91
#owner: https://api.github.com/users/elviskahoro

import sys
import anyio
import os
import dagger
import logging

from dagger import Container, dag

RELEASES_JSON_PATH: str = "src/generated/docs/platform-releases/releases.json"

DEFAULT_DOCKER_IMAGE_DIRECTORY: str = "dagger/docker-images"
DEFAULT_DOCKER_IMAGE_FILE_NAME: str = "python-3.11-buster.tar"
DEFAULT_DOCKER_IMAGE_ADDRESS: str = "python:3.11-slim-buster"

DEFAULT_USE_LOCAL_DOCKER_IMAGE: bool = True  # should be true by default
DEFAULT_EXPORT_DOCKER_IMAGE: bool = True

DEFAULT_CONNECTON_TIMEOUT: int = 3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


async def export_container_to_local_tarball(
    ctr: Container,
    image_file_path: str,
) -> str:
    file_path: str = await ctr.export(image_file_path)
    logger.info("Exported image filepath")
    return file_path


async def get_container_from_local_tarball(
    image_file_path: str,
) -> Container:
    logger.info("Pulling image from local tarball")
    return dagger.dag.container().import_(
        source=dag.host().file(
            path=image_file_path,
            no_cache=True,
        )
    )


async def get_container(
    local: bool,
) -> Container:
    logger.info("Getting container")
    image_file_path: str = (
        f"{DEFAULT_DOCKER_IMAGE_DIRECTORY}/{DEFAULT_DOCKER_IMAGE_FILE_NAME}"
    )
    if local:
        return await get_container_from_local_tarball(
            image_file_path=image_file_path,
        )

    logger.info("Pulling image from registry")
    return (
        dag.container()
        .from_(
            address=DEFAULT_DOCKER_IMAGE_ADDRESS,
        )
        .with_exec(
            args=[
                "pip",
                "install",
                "--upgrade",
                "pip",
            ],
        )
        .with_exec(
            args=[
                "pip",
                "install",
                "uv",
            ],
        )
    )


async def generate_release_notes(
    use_local_docker_image: bool = DEFAULT_USE_LOCAL_DOCKER_IMAGE,
    export_docker_image: bool = DEFAULT_EXPORT_DOCKER_IMAGE,
) -> None:
    async with dagger.connection(
        config=dagger.Config(
            log_output=sys.stderr,
            timeout=DEFAULT_CONNECTON_TIMEOUT,
        ),
    ):
        ctr: Container = await get_container(
            local=use_local_docker_image,
        )
        dir_src: Directory = dag.host().directory("./docs-frontend")
        pipeline_init: Container = ctr.with_mounted_directory(
            path="/docs-frontend",
            source=dir_src,
        ).with_workdir(
            path="/docs-frontend",
        )

        pipeline_with_libs: Container
        if not use_local_docker_image:
            pipeline_with_libs = pipeline_init.with_exec(
                args=[
                    "uv",
                    "pip",
                    "install",
                    "--system",
                    "-r",
                    "scripts/release_notes/requirements.txt",
                ],
            )

        else:
            pipeline_with_libs = pipeline_init

        if not use_local_docker_image and export_docker_image:
            _ = await export_container_to_local_tarball(
                ctr=pipeline_with_libs,
                image_file_path=f"{DEFAULT_DOCKER_IMAGE_DIRECTORY}/{DEFAULT_DOCKER_IMAGE_FILE_NAME}",
            )

        pipeline_run_llm: dagger.Container = (
            pipeline_with_libs.with_env_variable(
                name="OPENAI_API_KEY",
                value=os.environ["OPENAI_API_KEY"],
            )
            .with_env_variable(
                name="HYPERDX_API_KEY",
                value=os.environ["HYPERDX_API_KEY"],
            )
            .with_env_variable(
                name= "**********"
                value= "**********"
            )
            .with_exec(
                args=[
                    "python3",
                    "scripts/release_notes/runner.py",
                    "--model",
                    "openai",
                    "--concurrency",
                    "5",
                    "--release-notes-file-path",
                    "src/generated/docs/platform-releases/releases.json",
                ],
            )
        )
        std: str = await pipeline_run_llm.stdout()
        print(std)

        await pipeline_run_llm.export()

        # ++=========================================++=========================================++=========================================

        # releases_json_file: File = pipeline_install_requirements.file(
        #     path="/ws/docs-frontend/src/generated/docs/platform-releases/releases.json",
        # )
        # releases_json_file_content: str = await releases_json_file.contents()
        # print(releases_json_file_content)

        # with open(
        #     file="./docs-frontend/src/generated/docs/platform-releases/releases.json",
        #     mode="w",
        # ) as f:
        #     f.write(releases_json_file_content)
        #     f.flush()  # Ensure data is written to disk

        # with open(
        #     file="./docs-frontend/src/generated/docs/platform-releases/releases-elvis.json",
        #     mode="w",
        # ) as f:
        #     f.write(releases_json_file_content)
        #     f.flush()  # Ensure data is written to disk

        # ctr = (
        #     ctr.with_workdir(
        #         path="/docs-frontend",
        #     )
        #     .with_exec(
        #         args=["pip", "install", "uv"],
        #     )
        #     .with_exec(
        #         args=["uv", "pip", "install", "--system", *PIP_REQUIREMENTS],
        #     )
        # # )
        # logger.info(f"elvis::workdir: {workdir}")
        # paths: list[str] = await ctr.directory(path="/docs-frontend").glob("*")
        # print(paths)

        # from runner import PIP_REQUIREMENTS
        # from runner import main

        # print(await ctr.directory(path="/docs-frontend").glob("*"))
        # logger.info(f"elvis::ctr: {ctr}")
        # std: str = await ctr.stdout()
        # print(std)
        # await main(
        #     release_notes_file_path="docs-frontend/src/generated/docs/platform-releases/releases.json",
        #     concurrency=5,
        #     model="openai",
        # )

    # async with dagger.Connection(
    #     dagger.Config(
    #     ),
    # ) as client:
    #     dir_current: dagger.Directory = await client.host().directory(".")
    #     print(f"dir_current: {dir_current}")
    #     container_name: str = await dir_current.name()
    #     print(f"container_name: {container_name}")
    #     ctr: dagger.Container = (
    #         client.container()
    #         # .from_("python:3.11-slim-buster")
    #         .from_("python")
    #         .with_directory(
    #             path="/src",
    #             directory=dir_current,
    #         )
    #         .with_workdir(
    #             path="/src",
    #         )
    #         .with_exec(
    #             args=["pip", "install", "uv"],
    #         )
    #     )
    #     print(f"container: {ctr}")
    #     std: str = await ctr.stdout()
    #     print(f"std: {std}")

    # print(src_id)
    # src: dagger.Directory = client.host().directory(".")
    # ctr: dagger.Container = (
    #     client.container()
    #     .from_(
    #         address="python:3.11-slim-buster",
    #     )
    #     .with_directory(
    #         path="/ws",
    #         directory=src,
    #     )
    # )
    # name = await ctr.directory("/ws").name()
    # print(name)

    # # build and publish image
    # image_ref = "marvelousmlops/dagger_example:latest"
    # secret = "**********"
    #     name= "**********"
    #     plaintext= "**********"
    # )
    # build = (
    #     src.with_directory("/tmp/dist", client.host().directory("dist"))
    #     .docker_build(dockerfile="Dockerfile_dagger")
    #     .with_registry_auth(
    #         address=f"https://docker.io/{image_ref}",
    #         secret= "**********"
    #         username=os.environ["DOCKERHUB_USER"],
    #     )
    # )
    # await build.publish(f"{image_ref}")

    print("Elvis: Done")


if __name__ == "__main__":
    anyio.run(generate_release_notes)

    anyio.run(generate_release_notes)
